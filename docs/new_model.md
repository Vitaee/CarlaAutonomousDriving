graph TD
    subgraph "Input Processing"
        A[Center Camera Images] --> B[EfficientNet-B7]
        C[Left Camera Images] --> D[EfficientNet-B7]
        E[Right Camera Images] --> F[EfficientNet-B7]
        G[LiDAR Points] --> H[AdvancedLiDARProcessor]
    end
    
    subgraph "Feature Fusion"
        B --> I[AttentionFusion]
        D --> I
        F --> I
        H --> J[Feature Concatenation]
        I --> J
    end
    
    subgraph "Temporal Processing"
        J --> K[LSTM<br/>Temporal Consistency]
        K --> L[Feature Processor<br/>BatchNorm + Dropout]
    end
    
    subgraph "Multi-Task Outputs"
        L --> M[Steering Head<br/>Tanh Activation]
        L --> N[Speed Head<br/>Sigmoid Activation]
        L --> O[Emergency Brake<br/>Softmax Classifier]
    end
    
    subgraph "Safety Components"
        L --> P[SafetyModule]
        P --> Q[Uncertainty Estimation<br/>Softplus]
        P --> R[Anomaly Detection<br/>Reconstruction Loss]
    end
    
    subgraph "Safe Prediction"
        M --> S[predict_safe]
        Q --> S
        R --> S
        S --> T[Conservative Fallback<br/>When Uncertain]
    end
    
    style A fill:#e1f5fe
    style G fill:#e8f5e8
    style I fill:#fff3e0
    style K fill:#f3e5f5
    style P fill:#ffebee
    style S fill:#e8f5e8