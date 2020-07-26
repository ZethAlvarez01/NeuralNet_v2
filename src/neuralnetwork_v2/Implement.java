package neuralnetwork_v2;
import java.util.ArrayList;
import java.util.Stack;

/**
 *
 * @author Zeth
 */

public class Implement {
    ArrayList<Neural_layer> neural_net=new ArrayList<>();                       // Red neuronal                                            // Salida real para entrenamiento
    Matrix op=new Matrix();                                                     //Libreria de matrices hecha por yo
    
  
    public Implement(ArrayList<Neural_layer> neural_net){                       // Constructor 
        this.neural_net=neural_net;
    }
    
    /*
        Paso 1 Algortimo Feedforward 
        
        Metodo que se puede usar para prediccion de una serie de valores de
        entrada hacia la red.
    
    */
    
    public double[][] prediction(double[] xi){                                              
        double[][] x=new double[1][1];
        x[0]=xi;
        ArrayList<ArrayList<double[][]>> out=new ArrayList<>();                 // Arreglo OUT guarda todas las Z's y A's de cada capa
        ArrayList<double[][]> aux_out=new ArrayList<>();
        aux_out.add(x);
        aux_out.add(x);
        out.add(aux_out);
        

        for(int l=0;l<neural_net.size();l++){                                   // for que recorre toda la red
            ArrayList<double[][]> _out=new ArrayList<>();                       // Arrelo temporal que guarda Z y A de cada capa           
            double[][] z;                                                       // Arreglo de resultado de la suma ponderada Z
            double[][] producto=op.dot(x, neural_net.get(l).w);                 // Arreglo auxiliar para guardar el producto de Xn*Wm

            z=op.add(producto, neural_net.get(l).b);                            // Z = (Xn*Wm) + B
            _out.add(z);

            double[][] a=new double[1][z[0].length];                             // Arreglo para el resultado de la funcion de activacion
                                                          
            for(int i=0;i<z[0].length;i++){                                 
                a[0][i]=neural_net.get(l).act_f(z[0][i]);                       // Resultado Z pasado por la funcion de activacion A = F(Z)
            }
            _out.add(a);

            x=a;                                                               // A se convierte en la nueva entrada X
            out.add(_out);
        }     
        
        
        return out.get(out.size()-1).get(1);                                                              // Regresamos la X final 
                                                                                // (que seria la ultima A convertida en X por si existieran mas neuronas)
    }
    
    
    /*
        Paso 2 Algortimo Backpropagation
        
        Metodo que se usa para entrenar a la red con el algortimo de 
        backpropagation.
    
    */
 public double[][] train(double[] input,double[] target){                                              
        double[][] x=new double[1][];
        x[0]=input;
        double lr=0.09;                                                          // El ratio de aprendizaje es de 0.5
        
        ArrayList<double[][]> hidden_o=new ArrayList<>();
        hidden_o.add(x);
        int k=0;
        
        /*
            Se ejecuta el algortimo Feedforward pero guardando los valores
            de salida de cada capa.
            
            Se guardan en el arrayList 'hidden_o'
        */
        
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

        /*
            Arreglos auxiliares para los deltas
            (errores calculados en cada capa) 
        */
        
        double[] delta = null;
        double[][] delta_w;
        double[][] delta_w_h;
        double[][] w_aux = null;
        double[][] b_aux = null;
        
        double[][] delta_b;
        double[][] delta_b_h;
        
        
        for(int l=neural_net.size()-1;l>=0;l--){
            /*
                Calculo del error si se esta en la ultima capa
            */
            
            if(l==neural_net.size()-1){
                delta=new double[hidden_o.get(l+1)[0].length];
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    delta[i]=(hidden_o.get(l+1)[0][i]*(1-hidden_o.get(l+1)[0][i]))*(target[i]-hidden_o.get(l+1)[0][i]);
                }
                
                double[][] delta_x=new double[1][];
                delta_x[0]=delta;
                
                
                delta_w=op.scalar(op.dot(op.transpose(hidden_o.get(l)),delta_x),lr);
                delta_b=op.scalar(delta_x, lr);

                w_aux=new double[neural_net.get(l).w.length][neural_net.get(l).w[0].length];
                w_aux=neural_net.get(l).w;              
                
                b_aux=new double[neural_net.get(l).b.length][neural_net.get(l).b[0].length];
                b_aux=neural_net.get(l).b;
                
                neural_net.get(l).w=op.add(neural_net.get(l).w, delta_w);
                neural_net.get(l).b=op.add(neural_net.get(l).b, delta_b);

            }else{
                /*
                    Calculo del error si no es la ultima capa
                */
                double[] s_delta=new double[hidden_o.get(l+1)[0].length];
                
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    s_delta[i]=hidden_o.get(l+1)[0][i]*(1-hidden_o.get(l+1)[0][i]);
                }
                
                double[][] delta_x=new double[1][];
                delta_x[0]=delta;

                double[][] delta_h=op.dot(delta_x, op.transpose(w_aux));
                
                for(int i=0;i<hidden_o.get(l+1)[0].length;i++){
                    delta_h[0][i]=delta_h[0][i]*s_delta[i];
                }
                
                delta_w_h=op.scalar(op.dot(op.transpose(hidden_o.get(l)),delta_h),lr);
                delta_b_h=op.scalar(delta_h, lr);

                delta=delta_w_h[0];
                
                w_aux=new double[neural_net.get(l).w.length][neural_net.get(l).w[0].length];
                w_aux=neural_net.get(l).w;
                
                // Formulas para actualizar los nuevos pesos con base al error de la capa siguiente
                neural_net.get(l).w=op.add(neural_net.get(l).w, delta_w_h);

                double[][] b_aux_h=new double[neural_net.get(l).b.length][neural_net.get(l).b[0].length];
                b_aux_h=neural_net.get(l).b;
                
                // Formulas para actualizar los nuevos bias con base al error de la capa siguiente
                neural_net.get(l).b=op.add(b_aux_h,delta_b_h);

                b_aux=b_aux_h;
            }
        }
        return hidden_o.get(hidden_o.size()-1);                                                                 
    }
     
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

    private double[] cost_fuction(double[] yp, double[] yr, int tp) {
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
