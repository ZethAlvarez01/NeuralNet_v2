package neuralnetwork_v2;

import java.util.ArrayList;
/**
 *
 * @author Zeth
 */
public class train {
    ArrayList<Neural_layer> neural_net;
    ArrayList<double[][]> x=new ArrayList<>();
    double[] y;
    double lr;
    boolean train;

    
    public train(ArrayList<Neural_layer> neural_net,ArrayList<double[][]> x,double[] y,double lr,boolean train){
        this.neural_net=neural_net;
        for(int i=0;i<x.size();i++){
            this.x.add(x.get(i));
        }
        this.y=y;
        this.lr=lr;
        this.train=train;
    }
    

    public void training(){
        ArrayList<double[][]> Z_global=new ArrayList<>();
        ArrayList<double[][]> A_global=new ArrayList<>();
        
        Matrix op=new Matrix();
        System.out.println("Salida");
        for(int input=0;input<x.size();input++){
            for(int l=0;l<neural_net.size();l++){
                double[][] z;
                z=op.add(op.dot(x.get(input), neural_net.get(l).w), neural_net.get(l).b);
                Z_global.add(z);
                double[][] a=new double[1][z[0].length];
                for(int i=0;i<z[0].length;i++){
                    a[0][i]=neural_net.get(i).act_f(z[0][i]);
                }
                A_global.add(a);
                x.set(input, a);
            }            
            op.print(x.get(input));
        }
        System.out.println();
        System.out.println("A_global");
        if(train){
            for(int k=0;k<A_global.size();k++){
                op.print(A_global.get(k));
            }
        }

        
        
    }
    
  
    private double f_coste(double[] yp,double[] yr){
        double sum=0;
        for(int i=0;i<yp.length;i++){
            sum+=yp[i]-yr[i];
        }
        sum=Math.pow(sum/yp.length,2);
        return sum;
    }
}
