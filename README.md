# Clusterização pelo método hierárquico e K-means

![image](https://github.com/user-attachments/assets/47a8a2a8-b85c-4685-8ba6-86dece59aaff)

Kaggle: https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation

a) Método hierárquico

Gera o dendrograma, escolhe o número de clusters olhando uma linha horizontal (número de linhas verticais)
Parâmetros: Tipos de distância (euclidiana), método para comparar os clusters (complete = observação mais distante)
O dendrograma já cria cores diferentes sugerindo o número de clusters

b) Método K-means

Usando o método do elbow para determinar o número de clusters
estatística F mostra como cada atributo contribui para a formação do clusters
Se o nível de significância for menor que 0,05 (5%), sim o atributo é relevante para formação dos clusters
