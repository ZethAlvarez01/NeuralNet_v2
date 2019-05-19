package neuralnetwork_v2;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import javax.swing.JOptionPane;

/**
 *
 * @author Zeth
 */

public class NeuralNetwork_v2 {

    static ArrayList<Neural_layer> create_nn(int[] topology,int act_f){         // Crear la nueva red neuronal, recibe la topologia y el tipo de funcion de activacion que va a usar
       ArrayList<Neural_layer> nn=new ArrayList<>();                            // Arreglo de capas (red completa)
       
       for(int i=0;i<topology.length-1;i++){                                    // for del que recorre el arreglo de topologia
           Neural_layer layer=new Neural_layer(topology[i],topology[i+1],act_f);    // Creas una nueva capa con: (numero de neuronas de entrada, numero de salidas, tipo de funcion de activacion) 
           nn.add(layer);                                                           // se añade esa capa a la red completa
       }
       
       return nn;                                                               // Se regresa la red completa
    }

    /*
        Asi se deben declarar las cosas para que la red funcione correctamente:        
    */
    
    public static void main(String[] args) throws IOException {
        Matrix op=new Matrix();    
        
        /* 
            Se declara un arreglo de clases Neural_layer   
            Este arreglo sera nuestra red neuronal
        */        
        ArrayList<Neural_layer> neural_net;   
        
        /* 
            Un arreglo de enteros representa la topologia de nuestra red. 
            Cada posicion del arreglo es una capa en la red y el numero
            representa el numero de neuronas de dichas capas.
        
        Ejemplo: 
        
        topology es una topologia de 2 neuronas de entrada, 2 neuronas en una
        capa oculta y 1 neurona de salida.
        
        Esto para predecir las salidas de una compuerta logica XOR
        
        */
        int[] topology={2,2,1};                                                 
                                                                                                        
        double[][] set={{0,0},{1,0},{0,1},{1,1}};                               // Entradas de ejemplo binarias
        double[][] target={{0},{1},{1},{0}};                                    // Salidas correctas correspondientes a las entradas
        
        /*
            Se manda crear la nueva red neuronal 
        */
        neural_net=create_nn(topology,0);                                          

        
        /*
            Instancio el mi clase Implement que implementara la red o la 
            entrenara.
        */
        Implement aplicacion=new Implement(neural_net);
        double[][] output = null;
        
        //Se imprimen las entradas
        System.out.println("Entradas: ");
        op.print(set);
        System.out.println();
        
        //Se imprimen las salidas esperadas
        System.out.println("Salidas esperadas: ");
        op.print(target);
        System.out.println();
        
        /*
            Se ejecuta el metodo 'prediction' que pasa cada entrada atraves 
            de la red y da una salida para cada una de estas.
        */
        System.out.println("Salida sin entrenamiento: ");
        for(int j=0;j<4;j++){
            output=aplicacion.prediction(set[j]);
            op.print(output);
        }                  
        System.out.println();
        
        /*
            Como la red no esta entrenada los resultados de prediction son
            aleatorios por lo que procedemos a ejecutar el metodo 'train' 
            con 5000 iteraciones para las 4 entradas posibles y 4 salidas 
            correctas.
        */

        
        for(int i=0;i<5000;i++){
            for(int j=0;j<4;j++){
                aplicacion.train(set[j],target[j]);
            }           
        }
        
        /*
            Mostramos las salidas correspondientes a cada una de las 
            entradas pero de forma correcta.
        
        
            Para una sola salida correspondiente a una sola entrada solo
            es necesario pasarle el arreglo de datos de entrada del que 
            querramos conocer la salida.
        
            output=aplicacion.prediction(set[0]);
            op.print(output);
        
        */
        
       System.out.println("Salida con entrenamiento: ");
        for(int j=0;j<4;j++){
            output=aplicacion.prediction(set[j]);
            op.print(output);
        }
        System.out.println();
       
    }
}
