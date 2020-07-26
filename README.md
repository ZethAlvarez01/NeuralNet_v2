# Perceptrón multicapa

Perceptrón multicapa desarrollado en Java con ayuda del IDE NetBeans. Se puede modificar el tamaño de los parámetros de entrada, salida, así como el número de capas y neuronas intermedias.

## Instalar 

- Clonar el repositorio en tu equipo.

## Configuración

En el archivo **NeuralNetwork_v2** existen cuatro arreglos de tipo **double** que serán las entradas con las que alimentaremos a nuestra red, los cuales son: 

```
double[] entrada
```
Son los parámetros (entre 0 y 1) de entrada para evaluar y que la red nos arroje una predicción.

```
double[] salida
```
Es la salida real esperada para dicha entrada.

```
double[][] entradas
```
Es el arreglo de todas las entradas con las que se entrenara el sistema.

```
double[][] salidas
```
Es el arreglo de todas las salidas reales esperadas con las que se entrenara el sistema.

```
int[] topology={entrada.length,2,3,1};   
```
Es la topología que queremos para la red. En el ejemplo podemos ver una red con entradas n (tamaño del arreglo **entrada**), una capa oculta con 2 neuronas, una capa oculta de 3 neuronas y una salida de 1 neurona (Esta última debe coincidir con el número de elementos del arreglo **salida**).

Al ejecutar el programa (con ayuda del IDE) primero se mostrará en terminal una predicción con el arreglo **entrada**. Aquí la red NO esta entrenada.
```
double[][] Yp = imp.prediction(entrada);
        System.out.print("\n\Predicción SIN entrenamiento: ");
        op.print(Yp);
```

Después se procederá a entrenarla con el algoritmo de backpropagation. 10000 iteraciones, 0.09 de ratio de aprendizaje y usando todas las entradas y salidas de los arreglos **entradas** y **salidas**.
```
for(int i=0;i<10000;i++){
            //System.out.println("Iteracion: "+i);
            for(int j=0;j<entradas.length;j++){
               imp.train(entradas[j],salidas[j]); 
            }
        }
```
Al finalizar se mostrará en pantalla una nueva predicción usando el mismo arreglo **entrada** ya con la red entrenada.
```
Yp = imp.prediction(entrada);
        System.out.print("Predicción CON entrenamiento: ");
        op.print(Yp);
```

## NOTAS
La salida de la predicción así como del entrenamiento es un arreglo bidimensional del cual solo se ocupa el arreglo en la posición [0].


## Autor ✒️

* **Zeth Álvarez Hernández** - *Trabajo Inicial*
