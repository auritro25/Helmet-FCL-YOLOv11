# Helmet-FCL-YOLOv11

Research pipeline:
- YOLOv11s baseline
- Contrastive learning pretraining
- Flower federated learning (FedAvg/FedProx)
- External evaluation

Hardware target:
RTX 2060 6GB

Workflow:
1. Put Roboflow dataset in datasets/roboflow/
2. Put Mendeley dataset in datasets/edgevision/
3. Run preprocessing
4. Train baseline
5. Run contrastive pretraining
6. Run federated experiments
