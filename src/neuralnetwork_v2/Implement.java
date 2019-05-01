package neuralnetwork_v2;
import java.util.ArrayList;

/**
 *
 * @author Zeth
 */
public class Implement {
    ArrayList<Neural_layer> neural_net=new ArrayList<>();                       // Red neuronal
    double[][] x=new double[1][];                                               // Entradas iniciales(Inputs)
    
    public Implement(ArrayList<Neural_layer> neural_net,double[] x){            // Constructor 
        this.neural_net=neural_net;
        this.x[0]=x;
    }
    
    public double[][] Implement(){                                              
        Matrix op=new Matrix();                                                 // Mi libreria de operaciones con matrices
        
        for(int l=0;l<neural_net.size();l++){;                                  // for que recorre toda la red
                double[][] z;                                                   // Arreglo de resultado de la suma ponderada Z
                double[][] producto=op.dot(x, neural_net.get(l).w);             // Arreglo auxiliar para guardar el producto de Xn*Wm
                z=op.add(producto, neural_net.get(l).b);                        // Z = (Xn*Wm) + B
                double[][] a=new double[1][z[0].length];                        // Arreglo para el resultado de la funcion de activacion
                for(int i=0;i<z[0].length;i++){                                 
                    a[0][i]=neural_net.get(l).act_f(z[0][i]);                   // Resultado Z pasado por la funcion de activacion A = F(Z)
                }
                x=a;                                                            // A se convierte en la nueva entrada X
        }     
        
        return x;                                                               // Regresamos la X final 
                                                                                // (que seria la ultima A convertiva en X por si existieran mas neuronas)
    }
    
}
