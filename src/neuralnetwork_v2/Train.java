package neuralnetwork_v2;

import java.util.ArrayList;
/**
 *
 * @author Zeth
 */
public class Train {
    ArrayList<Neural_layer> neural_net;
    ArrayList<double[][]> x=new ArrayList<>();
    ArrayList<double[][]> y=new ArrayList<>();
    double lr;
    boolean train;

    
    public Train(ArrayList<Neural_layer> neural_net,ArrayList<double[][]> x,ArrayList<double[][]> y,double lr,boolean train){
        this.neural_net=neural_net;
        for(int i=0;i<x.size();i++){
            this.x.add(x.get(i));
        }
        for(int i=0;i<y.size();i++){
            this.y.add(y.get(i));
        }
        this.lr=lr;
        this.train=train;
    }
    

    public void training(){
        ArrayList<double[][]> Z_global=new ArrayList<>();
        ArrayList<double[][]> A_global=new ArrayList<>();
        
        Matrix op=new Matrix();
        
        for(int input=0;input<x.size();input++){
            for(int l=0;l<neural_net.size();l++){
                double[][] z;
                z=op.add(op.dot(x.get(input), neural_net.get(l).w), neural_net.get(l).b);
                Z_global.add(z);
                double[][] a=new double[1][z[0].length];
                for(int i=0;i<z[0].length;i++){
                    a[0][i]=neural_net.get(l).act_f(z[0][i]);
                }
                A_global.add(a);
                x.set(input, a);
            }            
           /*
            ArrayList<double[][]> delta=new ArrayList<>();
            if(train){
                for(int l=neural_net.size()-1;l>=0;l--){
                    double[][] a=A_global.get(l);
                    System.out.println(neural_net.get(l).n_conn+" "+neural_net.get(l).n_neuronas);
                    if(l==neural_net.size()-1){
                        System.out.println("Ultima capa");
                        //op.print(neural_net.get(l).w);
                        double[][] dev_f_costo=null;
                        for(int k=0;k<a[0].length;k++){
                            dev_f_costo=op.sub(a, y.get(input));
                        }
                        
                        double[][] dev_f_act=new double[1][a[0].length];
                        for(int k=0;k<a[0].length;k++){
                            dev_f_act[0][k]=neural_net.get(l).dev_act_f(a[0][k]);
                        }
                        
                        double[][] producto=new double[1][a[0].length];
                        for(int k=0;k<a[0].length;k++){
                            producto[0][k]=dev_f_costo[0][k]*dev_f_act[0][k];
                        }
                        
                        delta.add(0,producto);
                        
                    }else{
                        System.out.println("Hidden layers");
                        //op.print(neural_net.get(l).w);
                        //op.print(a);
                        
                        double[][] dev_f_act=new double[1][a[0].length];
                        for(int k=0;k<a[0].length;k++){
                            dev_f_act[0][k]=neural_net.get(l).dev_act_f(a[0][k]);
                        }
                        //op.print(dev_f_act);
                        
                        double[][] deltaA=delta.get(0);
                        
                        double[][] wA=op.transpose(neural_net.get(l+1).w);
                        //op.print(wA);
 
                        
                        double[][] res=op.dot(deltaA, wA);
                        //op.print(res);
                        
                        double[][] producto=new double[1][a[0].length];
                        for(int k=0;k<a[0].length;k++){
                            producto[0][k]=res[0][k]*dev_f_act[0][k];
                        }
                        
                        delta.add(0,producto);
                        
                    }
                    System.out.println(); 
                    
                    /////////////////////
                    //op.print(delta.get(0));
                    double[][] delta_lr=op.scalar(delta.get(0), lr);
                    neural_net.get(l).b=op.sub(neural_net.get(l).b, delta_lr);
                    //////////////////////
                    
                    /*for(int i=A_global.size()-1;i>=0;i--){
                        op.print(A_global.get(i));
                    }*/
                    /*
                    op.print(neural_net.get(l).w);
                    op.print(delta.get(0));
                    if(l-1==-1){
                        op.print(A_global.get(A_global.size()-1));
                    }else{
                       op.print(A_global.get(l-1)); 
                    }
                    
                    
                    
                } 
            }*/
            
            System.out.println();
            System.out.println("Salida");
            op.print(x.get(input));
            
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
