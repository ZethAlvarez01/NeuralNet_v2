package neuralnetwork_v2;

import java.util.ArrayList;

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

    public static void main(String[] args) {
        Matrix op=new Matrix();    
        ArrayList<Neural_layer> neural_net;   
          
        int[] topology={2,2,1};    
        
        double[][] set={{0,0},{1,0},{0,1},{1,1}};
        double[][] target={{0},{1},{1},{0}};
                                                   
        neural_net=create_nn(topology,0);                                       

        Implement entrenamiento=new Implement(neural_net);
        double[][] output = null;
        
        System.out.println("Entradas: ");
        op.print(set);
        System.out.println();
        
        System.out.println("Salidas esperadas: ");
        op.print(target);
        System.out.println();
        
        
        System.out.println("Salida sin entrenamiento: ");
        for(int j=0;j<4;j++){
            output=entrenamiento.prediction(set[j]);
            op.print(output);
        }                  
        System.out.println();
        
        /*
        System.out.println("Salida sin entrenamiento: (entrenamiento-prediccion)");
        output=entrenamiento.prediction(set[0]);
        op.print(output);
        */
        
//        System.out.println("Salida sin entrenamiento: (entrenamiento-train)");
//        output=entrenamiento.train(set[0], target[0]);
//        System.out.println();
//        op.print(output);
        
        for(int i=0;i<5000;i++){
            for(int j=0;j<4;j++){
                entrenamiento.train(set[j],target[j]);
            }           
        }
        
       System.out.println("Salida con entrenamiento: ");
        for(int j=0;j<4;j++){
            output=entrenamiento.prediction(set[j]);
            op.print(output);
        }
        System.out.println();

        
       
    }
}
