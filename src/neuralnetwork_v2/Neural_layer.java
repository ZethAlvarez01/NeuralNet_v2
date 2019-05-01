package neuralnetwork_v2;

public class Neural_layer {
    int n_conn;                 // Numero de conexiones (Entradas/Inputs)
    int n_neuronas;             // Numero de neuronas   (Salida/Outputs de la neurona)
    int act_f;                  // Funcion de activacion
    double b[][];               // Arreglo de BIAS (Sesgo)
    double w[][];               // Arreglo de pesos
    
    public Neural_layer(int n_conn,int n_neuronas,int act_f){                   // Creo una capa de las muchas de la red
        this.n_conn=n_conn;                                                     // Numero de conexiones (Entradas/Inputs)
        this.n_neuronas=n_neuronas;                                             // Numero de neuronas   (Salida/Outputs de la neurona)
        this.act_f=act_f;                                                       // Funcion de activacion
        this.b= new double[1][n_neuronas];                                      // Declaracion de arreglo de BIAS
        
        for(int i=0;i<n_neuronas;i++){                                          // Se rellena el arreglo de BIAS con aleatorios
            b[0][i]=Math.random()*2-1;                                          // entre 0 y 1 
        }                       
        
        this.w= new double[n_conn][n_neuronas];                                 // Declaracion de arreglo de pesos [Entradas]x[Salidas] 
        
        for(int i=0;i<n_conn;i++){                                              // Se rellena el arreglo de pesos con aleatorios
            for(int j=0;j<n_neuronas;j++){                                      // entre 0 y 1
                  w[i][j]=Math.random()*2-1;
            }
        }
    }
    
    // Funciones de activacion
    /*
        0 : Sigmoide
        1 : ReLu
        2 : Tangente hiperbolica
    
    */
    
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
    
    
    // Derivada de las funciones de activacion
    /*
        0 : Derivada de Sigmoide
        1 : Derivada de ReLu
        2 : Derivada de Tangente hiperbolica
    
    */
    
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
