#[derive(Debug, Clone, Default)]
pub struct IngestTimings {
    pub per_vector_ns: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingSummary {
    pub total_ns: u64,
    pub mean_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub vectors: u64,
}

impl TimingSummary {
    pub fn format_duration(ns: u64) -> String {
        if ns >= 1_000_000_000 {
            format!("{:.3}s", ns as f64 / 1_000_000_000.0)
        } else if ns >= 1_000_000 {
            format!("{:.3}ms", ns as f64 / 1_000_000.0)
        } else if ns >= 1_000 {
            format!("{:.3}µs", ns as f64 / 1_000.0)
        } else {
            format!("{ns}ns")
        }
    }
}

impl IngestTimings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, elapsed_ns: u64) {
        self.per_vector_ns.push(elapsed_ns);
    }

    pub fn summary(&self) -> TimingSummary {
        if self.per_vector_ns.is_empty() {
            return TimingSummary {
                total_ns: 0,
                mean_ns: 0.0,
                min_ns: 0,
                max_ns: 0,
                vectors: 0,
            };
        }

        let mut total_ns = 0_u64;
        let mut min_ns = u64::MAX;
        let mut max_ns = 0_u64;

        for &ns in &self.per_vector_ns {
            total_ns += ns;
            min_ns = min_ns.min(ns);
            max_ns = max_ns.max(ns);
        }

        let vectors = self.per_vector_ns.len() as u64;
        let mean_ns = total_ns as f64 / vectors as f64;

        TimingSummary {
            total_ns,
            mean_ns,
            min_ns,
            max_ns,
            vectors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_stats() {
        let mut timings = IngestTimings::new();
        timings.record(100);
        timings.record(200);
        timings.record(300);

        let summary = timings.summary();
        assert_eq!(summary.total_ns, 600);
        assert_eq!(summary.mean_ns, 200.0);
        assert_eq!(summary.min_ns, 100);
        assert_eq!(summary.max_ns, 300);
        assert_eq!(summary.vectors, 3);
    }

    #[test]
    fn empty_summary() {
        let timings = IngestTimings::new();
        let summary = timings.summary();
        assert_eq!(summary.vectors, 0);
        assert_eq!(summary.total_ns, 0);
    }
}
