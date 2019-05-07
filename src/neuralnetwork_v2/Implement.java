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
        double lr=0.5;
        
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

        //hidden_o.remove(0);
        double[] delta = null;
        double[][] delta_w;
        double[][] delta_w_h;
        double[][] w_aux = null;
        double[][] b_aux = null;
        
        double[][] delta_b;
        double[][] delta_b_h;
        
        
        for(int l=neural_net.size()-1;l>=0;l--){
             
            
            if(l==neural_net.size()-1){
                
                //System.out.println("Capa: "+l+"\n");
                
                //op.print(hidden_o.get(l+1));
                delta=new double[hidden_o.get(l+1)[0].length];
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    delta[i]=(hidden_o.get(l+1)[0][i]*(1-hidden_o.get(l+1)[0][i]))*(target[i]-hidden_o.get(l+1)[0][i]);
                    //System.out.print(delta[i]+" ");
                   //System.out.println(hidden_o.get(l+1)[0][i]*(1-hidden_o.get(l+1)[0][i]));
                }
                //System.out.println();
                
                double[][] delta_x=new double[1][];
                delta_x[0]=delta;
                
                
                //op.print(hidden_o.get(l));
                
                delta_w=op.scalar(op.dot(op.transpose(delta_x), hidden_o.get(l)),lr);
                delta_b=op.scalar(delta_x, lr);
                //System.out.println("delta_w");
                //op.print(delta_w);
                //op.print(delta_b);
                //System.out.println("w");
                w_aux=new double[neural_net.get(l).w.length][neural_net.get(l).w[0].length];
                w_aux=neural_net.get(l).w;
                
                //op.print(w_aux);
                
                b_aux=new double[neural_net.get(l).b.length][neural_net.get(l).b[0].length];
                b_aux=neural_net.get(l).b;
                
                neural_net.get(l).w=op.add(neural_net.get(l).w, op.transpose(delta_w));
                
                //op.print(neural_net.get(l).w);
                
                neural_net.get(l).b=op.add(neural_net.get(l).b, delta_b);
                
                //System.out.println("\nFin de la capa: "+l+"\n");
            }else{
                //System.out.println("Capa: "+l+"\n");
                
                //System.out.println("vieja w");
                //op.print(w_aux);
                double[] s_delta=new double[hidden_o.get(l+1)[0].length];
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    s_delta[i]=hidden_o.get(l+1)[0][i]*(1-hidden_o.get(l+1)[0][i]);
                    //System.out.println(s_delta[i]);
                }
                //System.out.println();
                
                double[][] delta_x=new double[1][];
                delta_x[0]=delta;

                double[][] delta_h=op.dot(delta_x, op.transpose(w_aux));
                
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    delta_h[0][i]=delta_h[0][i]*s_delta[i];
                }
                
                delta_w_h=op.scalar(op.dot(op.transpose(delta_h), hidden_o.get(l)),lr);
                delta_b_h=op.scalar(delta_h, lr);

                delta=delta_w_h[0];
                
                w_aux=new double[neural_net.get(l).w.length][neural_net.get(l).w[0].length];
                w_aux=neural_net.get(l).w;
                
                neural_net.get(l).w=op.add(neural_net.get(l).w, op.transpose(delta_w_h));

                double[][] b_aux_h=new double[neural_net.get(l).b.length][neural_net.get(l).b[0].length];
                b_aux_h=neural_net.get(l).b;
                
                neural_net.get(l).b=op.add(b_aux_h,delta_b_h);

                
                b_aux=b_aux_h;

                //System.out.println("\nFin de la capa: "+l);
            }
            
            
            
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
