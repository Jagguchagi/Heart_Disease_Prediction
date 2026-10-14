# Submission Evidence Checklist

Verified implementation/results are summarized in the bundle's top-level `README.md`. Save genuine screenshots from the lab VM or GitHub here before submission; do not count plot exports as infrastructure screenshots.

- [x] EDA and model-evaluation plots copied here: `class_balance.jpeg`, `correlation_heatmap.jpeg`, `feature_distributions.jpeg`, `logistic_regression_confusion_matrix.jpeg`, `logistic_regression_roc_curve.jpeg`, `random_forest_confusion_matrix.jpeg`, `random_forest_roc_curve.jpeg`. Source figures are also under `../artifacts/`.
- [ ] MLflow UI showing both completed model runs, metrics, and artifacts.
- [ ] Successful GitHub Actions run summary and job stages.
- [ ] Docker build and running Compose containers.
- [ ] Container API `/health` and `/predict` responses.
- [ ] Prometheus API target showing UP.
- [x] Grafana dashboard showing request and latency metrics: `grafana-dashboard.png`.
- [x] Kubernetes Ready pod and LoadBalancer service: `kubernetes-deployment.png`.
- [x] Successful Kubernetes LoadBalancer `/health` and `/predict` responses: `kubernetes-deployment.png`.
- [ ] Short end-to-end demo video and final ten-page report (listed here for submission tracking; store the video/report at the course-required location).
