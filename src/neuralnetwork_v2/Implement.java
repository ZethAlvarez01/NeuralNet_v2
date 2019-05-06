package neuralnetwork_v2;
import java.util.ArrayList;
import java.util.Stack;

/**
 *
 * @author Zeth
 */
public class Implement {
    ArrayList<Neural_layer> neural_net=new ArrayList<>();                       // Red neuronal                                            // Salida real para entrenamiento
    Matrix op=new Matrix();
    
    public Implement(ArrayList<Neural_layer> neural_net){            // Constructor 
        this.neural_net=neural_net;
    }
    
    public double[][] prediction(double[] xi){                                              
        double[][] x=new double[1][];
        x[0]=xi;
        
        for(int l=0;l<neural_net.size();l++){                                   // for que recorre toda la red
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
    
     public double[][] train(double[] input,double[] target){                                              
        double[][] x=new double[1][];
        x[0]=input;
        double lr=0.05;
        
        ArrayList<double[][]> hidden_o=new ArrayList<>();
        hidden_o.add(x);
        int k=0;
        for(int l=0;l<neural_net.size();l++){                                   
                double[][] z;                                                   
                double[][] producto=op.dot(hidden_o.get(k), neural_net.get(l).w);            
                z=op.add(producto, neural_net.get(l).b);                        
                double[][] a=new double[1][z[0].length];                        
                for(int i=0;i<z[0].length;i++){                                 
                    a[0][i]=neural_net.get(l).act_f(z[0][i]);                   
                }
                hidden_o.add(a);
                k++;
        }     
        
        double[][] delta_w;
        double[][] delta_b;
        double[][] error = null;
        double[][] s;
        double[][] gradiente;
        
        hidden_o.remove(0);
        
        
        for(int l=neural_net.size()-1;l>=0;l--){

                    //Error ultima capa target-output
                    error=new double[1][hidden_o.get(l)[0].length];
                    for(int i=0;i<target.length;i++){
                        error[0][i]=target[i]-hidden_o.get(l)[0][i];
                    }

                    //op.print(error);

                    //Derivada de la sigmoide
                    s=new double[1][hidden_o.get(l)[0].length];
                    for(int i=0;i<target.length;i++){
                        s[0][i]=hidden_o.get(l)[0][i]*(1-hidden_o.get(l)[0][i]);
                    }

                    //Calculo del gradiente
                    gradiente=op.dot(op.transpose(s), error);
                    gradiente=op.scalar(gradiente, lr);

                    //Calculo de la delta W
                    double[][] hidden_T=op.transpose(neural_net.get(l).w);
                    delta_w=op.dot(gradiente, hidden_T);

                    //Actualizacio de pesos
                    neural_net.get(l).w=op.add(neural_net.get(l).w, op.transpose(delta_w));

                    //Calculo de la delta BIAS
                    delta_b=op.dot(gradiente, op.transpose(neural_net.get(l).b));

                    //Actualizacio del BIAS
                    neural_net.get(l).b=op.add(neural_net.get(l).b, op.transpose(delta_b));
               
        }
        
        

        return hidden_o.get(hidden_o.size()-1);                                                                  
    }
    
    
    // Funcion de coste y su derivada
    /*
        0 : Minimos cuadrados ordinarios
        1 : Devivada de la funcion de minimos cuadrados ordinarios
    
    */
    
    public double[] cost_function(double[] yp,double[] yr,int tp){
        double[] cost=new double[yp.length];
        double aux=0;
        
        switch(tp){
            case 0:
                for(int i=0;i<yp.length;i++){
                    cost[i]=Math.pow((yp[i]-yr[i]), 2);
                }
                
                for(double mean : cost){
                    aux += mean;
                }
                
                cost[0]=aux/cost.length;
                
                return cost;
                
            case 1:
                
                for(int i=0;i<yp.length;i++){
                    cost[i]=yp[i]-yr[i];
                }
                
                return cost;
                
            default:
                return cost;
        }
    }
}
