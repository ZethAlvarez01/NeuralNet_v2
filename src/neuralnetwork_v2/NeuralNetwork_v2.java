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

    public static void main(String[] args) throws IOException {
        Matrix op=new Matrix();  
        double entrada[] = {0,1,0,1,0};
        double salida[] = {1,0};
        double lr = 0.5;
        
        /* 
            Un arreglo de enteros representa la topologia de nuestra red. 
            Cada posicion del arreglo es una capa en la red y el numero
            representa el numero de neuronas de dichas capas.
        */
        int[] topology={entrada.length,8,4,7,2};                                                 
                                                                                                        
        
        /* 
            Se declara un arreglo de clases Neural_layer   
            Este arreglo sera nuestra red neuronal
        */        
        ArrayList<Neural_layer> neural_net=new ArrayList<>();   
        
        /*
            Se manda crear la nueva red neuronal 
            Se manda la tipologia y la funcion de activacion
        */
        for(int i=0;i<topology.length-1;i++){
            Neural_layer l = new Neural_layer(topology[i],topology[i+1],0);
            neural_net.add(l);
        }                                    

        for(int i=0;i<neural_net.size();i++){
            op.print(neural_net.get(i).w);
            System.out.println();
            op.print(neural_net.get(i).b);
            System.out.println();
        }
        
        //Implementacion para prediccion
        Implement imp=new Implement(neural_net);
        
        //Yp es el resultado de la prediccion
        double[][] Yp = imp.prediction(entrada);
        System.out.print("\n\nPrediccion: ");
        op.print(Yp);
        
        double cF = imp.cost_function(Yp,salida);
        System.out.println("Error: "+cF);
        
        //Entrenamiento
        imp.train(entrada,salida,lr);
        
       
    }
}
