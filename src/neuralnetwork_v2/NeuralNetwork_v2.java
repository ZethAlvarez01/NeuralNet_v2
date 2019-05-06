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
        ArrayList<Neural_layer> neural_net;                                     // Instancio una nueva red neuronal    
        
        
          
        int[] topology={2,2,1};                                                 // Topologia de la nueva red     (Input, Hidden layers [n capas], Output)
        
        Matrix op=new Matrix();                                                 // Mi bonita libreria de operaciones con matrices
                                                   
        neural_net=create_nn(topology,0);                                       // Creas la nueva red pasandole como parametros (Topologia, tipo de funcion de activacion) 
        /*
        //Pruebas con xor
        //Asignacion de pesos 
        
        //Capa 1
        double[][] l1_w={{-6.026713144652567,-4.777204633354563},{6.158038787679284,5.065332844327768}};
        double[][] l1_b={{-3.5062464713538244,2.3254362447002053}};
        
        //Capa 2
        double[][] l2_w={{7.620068825646342},{-7.411676355786206}};
        double[][] l2_b={{3.46980173186714}};
        
        //Le apaso los pesos actualizados
        neural_net.get(0).w=l1_w;
        neural_net.get(0).b=l1_b;
        
        neural_net.get(1).w=l2_w;
        neural_net.get(1).b=l2_b;
        
        ////////////////////////
        */
        //Implementación 
        double[] x={0,0};
  
        Implement implementacion=new Implement(neural_net);                              // Implementacion paso la red completa y los datos de entrada
        double[][] output=implementacion.prediction(x);                                      // Resultado
        
        //Imprimo la entrada

        System.out.println("Entrada: ");
        double[][] xa=new double[1][];
        xa[0]=x;
        op.print(xa);
        
        //Imprimo la salida
        
        System.out.println("Salida sin entrenar: ");
        op.print(output);

        
        
        // Fase de entrenamiento
        Implement entrenamiento=new Implement(neural_net);
        
        double[][] set={{0,0},{0,1},{1,0},{1,1}};
        double[][] target={{0},{1},{1},{0}};
        
        for(int i=0;i<2500;i++){
            int rand=(int)(Math.random()*4);
            entrenamiento.train(set[rand],target[rand]);
        }

       System.out.println("Salida con entrenamiento: ");
       output=entrenamiento.prediction(x);
       op.print(output);
       
    }
}
