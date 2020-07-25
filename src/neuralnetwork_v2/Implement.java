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
    
    public double cost_function(double[][] a,double[] salida){
        double cost = 0.0;
        for(int i=0;i<a[0].length;i++){
            cost += Math.pow(a[0][i] - salida[i],2);
        }
        return cost/a[0].length;
    }
    
    public double[] derivada_cost_function(double[][] a,double[] salida){
        double[] cost = new double[a[0].length];
        
        for(int i=0;i<a[0].length;i++){
            cost[i] = a[0][i] - salida[i];
        }
        return cost;
    }
  
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
        
        System.out.println();
        
        for(int i=0;i<out.size();i++){
            System.out.println("Z");
            op.print(out.get(i).get(0));
            System.out.println("A");
            op.print(out.get(i).get(1));
        }
        
        
        return out.get(out.size()-1).get(1);                                                              // Regresamos la X final 
                                                                                // (que seria la ultima A convertida en X por si existieran mas neuronas)
    }
    
    
    /*
        Paso 2 Algortimo Backpropagation
        
        Metodo que se usa para entrenar a la red con el algortimo de 
        backpropagation.
    
    */
    
    public void train(double[] xi,double[] yi,double lr){ 
        System.out.println();
        System.out.println();
        double[][] x=new double[1][1];
        x[0]=xi;
        double[][] y=new double[1][1];
        y[0]=yi;
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
        double[][] _W={{0,0}};
        ArrayList<double[]> delta = new ArrayList<>();
        
       for(int l=neural_net.size()-1;l>=0;l--){
           System.out.println("L = "+l);
           double[][] z=out.get(l+1).get(0);
           double[][] a=out.get(l+1).get(1);
           
            if(l==(neural_net.size()-1)){
               /*ULTIMA CAPA*/
               //Calculo de la delta
               double[] _delta1 = derivada_cost_function(a,y[0]);
               for(int i=0;i<_delta1.length;i++){
                   _delta1[i] = _delta1[i] * neural_net.get(l).dev_act_f(a[0][i]);
               }
               delta.add(_delta1);
               
            

            }else{
               //Capas L-n
               //Calculo de la delta
               double[][] aux_delta = new double[1][delta.get(0).length];
               aux_delta[0] = delta.get(0);        
               double[][] _delta2 = op.dot(aux_delta, op.transpose(_W));
               
               for(int i=0;i<_delta2[0].length;i++){
                   _delta2[0][i] = _delta2[0][i] * neural_net.get(l).dev_act_f(a[0][i]);
               }

               delta.add(0,_delta2[0]);
               
            }
            
            //Fin del calculod de deltas
             _W = neural_net.get(l).w;
                
            //Actualizacion de W
            double[][] out_l = op.transpose(out.get(l).get(1));
            double[][] aux_delta = new double[1][delta.get(0).length];
            aux_delta[0] = delta.get(0);
            double[][] out_1_por_delta = op.dot(out_l, aux_delta);

            neural_net.get(l).w = op.sub(neural_net.get(l).w,out_1_por_delta);
            neural_net.get(l).w = op.scalar(neural_net.get(l).w, lr);

            //Actualizacion de B
            double prom_deltas = 0.0;

            for(int i=0;i<delta.get(0).length;i++){
                prom_deltas += delta.get(0)[i];
            }
            prom_deltas = prom_deltas / delta.get(0).length;

            for(int i=0;i<neural_net.get(l).b[0].length;i++){
                neural_net.get(l).b[0][i] = neural_net.get(l).b[0][i] - prom_deltas;
            }

            neural_net.get(l).b  = op.scalar(neural_net.get(l).b, lr);
            
       }
        

        
        
    }
    
    
    
}
