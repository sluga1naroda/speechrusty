//! Priority job queue for the recognition and translation workers.
//!
//! Final jobs (and cancellations) go in arrival order and always before
//! interim ones; of the interim ones the freshest snapshot of each utterance is taken,
//! stale ones and those already "closed" by a final are dropped.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Condvar, Mutex};

pub trait QueueItem {
    /// Utterance id.
    fn key(&self) -> u64;
    fn is_interim(&self) -> bool;
}

pub struct PriorityQueue<T> {
    inner: Mutex<(VecDeque<T>, bool)>,
    cv: Condvar,
}

impl<T> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self { inner: Mutex::new((VecDeque::new(), false)), cv: Condvar::new() }
    }
}

impl<T: QueueItem> PriorityQueue<T> {
    pub fn push(&self, item: T) {
        let mut g = self.inner.lock().expect("queue poisoned");
        g.0.push_back(item);
        self.cv.notify_one();
    }

    /// After closing, the worker drains what is left and then gets `None`.
    pub fn close(&self) {
        self.inner.lock().expect("queue poisoned").1 = true;
        self.cv.notify_all();
    }

    pub fn next(&self) -> Option<T> {
        let mut g = self.inner.lock().expect("queue poisoned");
        loop {
            if let Some(item) = pick(&mut g.0) {
                return Some(item);
            }
            if g.1 {
                return None;
            }
            g = self.cv.wait(g).expect("queue poisoned");
        }
    }

    /// Whether the queue holds a final job for the utterance (its interim result is no longer needed).
    pub fn has_final_for(&self, key: u64) -> bool {
        self.inner.lock().expect("queue poisoned").0.iter().any(|j| j.key() == key && !j.is_interim())
    }
}

fn pick<T: QueueItem>(q: &mut VecDeque<T>) -> Option<T> {
    let closed: HashSet<u64> = q.iter().filter(|j| !j.is_interim()).map(|j| j.key()).collect();
    let mut latest: HashMap<u64, usize> = HashMap::new();
    for (i, j) in q.iter().enumerate() {
        if j.is_interim() {
            latest.insert(j.key(), i);
        }
    }
    let mut idx = 0;
    q.retain(|j| {
        let keep = !j.is_interim() || (!closed.contains(&j.key()) && latest.get(&j.key()) == Some(&idx));
        idx += 1;
        keep
    });
    if let Some(pos) = q.iter().position(|j| !j.is_interim()) {
        return q.remove(pos);
    }
    q.pop_back()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct J(u64, bool, u32);
    impl QueueItem for J {
        fn key(&self) -> u64 {
            self.0
        }
        fn is_interim(&self) -> bool {
            self.1
        }
    }

    #[test]
    fn finals_first_then_latest_interim() {
        let q = PriorityQueue::default();
        q.push(J(1, true, 1));
        q.push(J(1, true, 2));
        q.push(J(2, true, 1));
        q.push(J(3, false, 0));
        q.push(J(2, true, 5));
        assert_eq!(q.next(), Some(J(3, false, 0)));
        assert_eq!(q.next(), Some(J(2, true, 5)));
        assert_eq!(q.next(), Some(J(1, true, 2)));
        q.close();
        assert_eq!(q.next(), None);
    }

    #[test]
    fn interim_dropped_when_final_queued() {
        let q = PriorityQueue::default();
        q.push(J(1, true, 3));
        q.push(J(1, false, 0));
        q.push(J(2, true, 3));
        assert!(q.has_final_for(1) && !q.has_final_for(2));
        q.push(J(2, false, 0));
        q.close();
        let got: Vec<_> = std::iter::from_fn(|| q.next()).collect();
        assert_eq!(got, vec![J(1, false, 0), J(2, false, 0)]);
    }
}
