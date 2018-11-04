package neuralnetwork_v2;
import java.util.ArrayList;

/**
 *
 * @author Zeth
 */
public class Implement {
    ArrayList<Neural_layer> neural_net=new ArrayList<>();
    double[][] x=new double[1][];
    
    public Implement(ArrayList<Neural_layer> neural_net,double[] x){
        this.neural_net=neural_net;
        this.x[0]=x;
    }
    
    public double[][] Implement(){
        double[][] output=null;
        Matrix op=new Matrix();
        
        for(int l=0;l<neural_net.size();l++){;
                double[][] z;
                double[][] producto=op.dot(x, neural_net.get(l).w);
                z=op.add(producto, neural_net.get(l).b);
                double[][] a=new double[1][z[0].length];
                for(int i=0;i<z[0].length;i++){
                    a[0][i]=neural_net.get(l).act_f(z[0][i]);
                }
                x=a;
        }     
        
        return x;
    }
    
}
