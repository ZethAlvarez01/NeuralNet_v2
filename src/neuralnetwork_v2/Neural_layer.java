package neuralnetwork_v2;

public class Neural_layer {
    int n_conn;
    int n_neuronas;
    int act_f;
    double b[][];
    double w[][];
    
    public Neural_layer(int n_conn,int n_neuronas,int act_f){
        this.n_conn=n_conn;
        this.n_neuronas=n_neuronas;
        this.act_f=act_f;
        this.b= new double[1][n_neuronas];
        for(int i=0;i<n_neuronas;i++){
            b[0][i]=Math.random()*2-1;
        }
        this.w= new double[n_conn][n_neuronas];
        for(int i=0;i<n_conn;i++){
            for(int j=0;j<n_neuronas;j++){
                  w[i][j]=Math.random()*2-1;
            }
        }
        
        
    }

    Neural_layer() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
    public double act_f(double x){
        switch(act_f){
            case 0:
                return 1/(1+Math.exp(-x));
            case 1:
                return Math.max(0, x);
            case 2: 
                return Math.tanh(x);
        }
        return 0;
    }
    
    public double dev_act_f(double x){
         switch(act_f){
            case 0:
                return x*(1-x);
            case 1:
                if(x<0){
                    return 0;
                }else if(x>=0){
                    return 1;
                }
                break;
            case 2: 
                return 1-Math.pow(Math.tanh(x),2);
        }
         return 0;
    }
}
