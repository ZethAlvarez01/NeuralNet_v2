package neuralnetwork_v2;

import java.util.ArrayList;

/**
 *
 * @author Zeth
 */
public class NeuralNetwork_v2 {
    
    static ArrayList<Neural_layer> create_nn(int[] topology,int act_f){
       ArrayList<Neural_layer> nn=new ArrayList<>();
       
       for(int i=0;i<topology.length-1;i++){
           Neural_layer l=new Neural_layer(topology[i],topology[i+1],act_f);
           nn.add(l);
       }
       
       return nn;
    }

    public static void main(String[] args) {
        ArrayList<Neural_layer> neural_net;
        
        int[] topology={2,2,3,1};
        
        Matrix op=new Matrix();
        
        ArrayList<double[][]> x=new ArrayList<>();
        
        double[][] aux0={{0,0}};
        double[][] aux1={{0,1}};
        double[][] aux2={{1,0}};
        double[][] aux3={{1,1}};
        
        x.add(aux0);
        x.add(aux1);
        x.add(aux2);
        x.add(aux3);
        
        double[] y={0,1,1,0};
        
        neural_net=create_nn(topology,0);
        
        for(int i=0;i<neural_net.size();i++){
            System.out.println("Layer "+(i+1));
            System.out.println(neural_net.get(i).n_conn+" "+neural_net.get(i).n_neuronas);
            System.out.println("b");
            for(int j=0;j<neural_net.get(i).b.length;j++){
                for(int k=0;k<neural_net.get(i).b[0].length;k++){
                    System.out.println(neural_net.get(i).b[j][k]);
                }
            }
            System.out.println("w");
            for(int k=0;k<neural_net.get(i).n_conn;k++){
                for(int j=0;j<neural_net.get(i).n_neuronas;j++){
                      System.out.print(neural_net.get(i).w[k][j]+"  ");
                }
                System.out.println();
            }
            System.out.println();
        }
        
        /*XOR test*/
        /*
        double[][] l1_w={{-6.026713144652567,-4.777204633354563},{6.158038787679284,5.065332844327768}};
        double[][] l1_b={{-3.5062464713538244,2.3254362447002053}};
        
        double[][] l2_w={{7.620068825646342},{-7.411676355786206}};
        double[][] l2_b={{3.46980173186714}};
        
        neural_net.get(0).w=l1_w;
        neural_net.get(0).b=l1_b;
        
        neural_net.get(1).w=l2_w;
        neural_net.get(1).b=l2_b;
        */
        train train=new train(neural_net,x,y,0.5,true);
        train.training();
        
        System.out.println();
        
        //Forward pass
        ArrayList<double[][]> out=new ArrayList<>();
        out.add(x.get(1));
        
        train eje=new train(neural_net,out,y,0.5,false);
        eje.training();
        
        

    }
}
