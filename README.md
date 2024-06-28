We introduced a novel framework for relation extraction using Graph Neural Networks (GNNs), focusing on enhancing model performance through contrastive learning. Our approach converts textual data into graphs where nodes represent tokens, enriched with Part-of-Speech (POS) and Named Entity Recognition (NER) tags encoded as one-hot vectors. This structure enables the capture of complex linguistic features within sentences.

Utilizing a Graph Attention Network (GAT) with multiple GATConv layers, our model adaptively aggregates contextual information from the nodes' neighborhoods. A global mean pooling layer then consolidates these features into a unified graph-level representation for predicting relations.

Central to our methodology is the application of contrastive learning, which refines embedding spaces by training the model to minimize distances between embeddings of graph pairs with similar relations and maximize those with different relations. This strategy, combined with relation classification, forms a dual-optimization process, enhancing the model's ability to discriminate between relations.

Training iterates over graph pairs, optimizing a combined loss function addressing both contrastive learning and classification objectives. We assess the model's accuracy, precision, recall, and F1 score on a test dataset, demonstrating its effectiveness in relation extraction.
