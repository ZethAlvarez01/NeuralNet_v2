package neuralnetwork_v2;
/**
 *
 * @author Zeth
 */
public class Matrix {
    
    //Sum of matrices DOUBLE + DOUBLE
    public double[][] add(double[][] a,double[][] b){
        if((a.length != b.length) || (a[0].length != b[0].length)){
            System.out.println("Error: Matrices with different order");
            return null;
        }else{
            double[][] res =new double[a.length][a[0].length];
            for(int i=0;i<a.length;i++){
                for(int j=0;j<a[i].length;j++){
                    res[i][j]=a[i][j]+b[i][j];
                }
            }
            return res;
        }
    }
    
    //Subtraction of matrices DOUBLE - DOUBLE
    public double[][] sub(double[][] a,double[][] b){
        if((a.length != b.length) || (a[0].length != b[0].length)){
            System.out.println("SUB Error: Matrices with different order");
            return null;
        }else{
            double[][] res =new double[a.length][a[0].length];
            for(int i=0;i<a.length;i++){
                for(int j=0;j<a[i].length;j++){
                    res[i][j]=a[i][j]-b[i][j];
                }
            }
            return res;
        }
    }
    
    //Scalar product of a matrix DOUBLE * DOUBLE
    public double[][] scalar(double[][] a,double s){
        double[][] res =new double[a.length][a[0].length];
        for(int i=0;i<a.length;i++){
            for(int j=0;j<a[i].length;j++){
                res[i][j]=a[i][j]*s;
            }
        }
        return res;
   }
   
   //Product matrix DOUBLE * DOUBLE
   public double[][] dot(double[][] a, double[][] b){
       if(a[0].length != b.length){
           System.out.println("Error: Column of matrix A "+a[0].length+" is different from row "
                            + "of matrix B "+b.length);
           return null;
       }else{
        double[][] res=new double[a.length][b[0].length];
        
        for(int i=0;i<res.length;i++){
            for(int j=0;j<res[0].length;j++){
                double sum=0;
                for(int k=0;k<a[0].length;k++){
                    sum+=a[i][k]*b[k][j];
                }
                res[i][j]=sum;
            }
        }
        
        return res;
       }
   }
   
   //Transposed DOUBLE
   public double[][] transpose(double[][] a){
       double[][] res=new double[a[0].length][a.length];
       
       for(int i=0;i<res.length;i++){
           for(int j=0;j<res[0].length;j++){
               res[i][j]=a[j][i];
           }
       }
       
       return res;
   }
   
   //Print matrix DOUBLE
   public void print(double[][] a){
       for(int i=0;i<a.length;i++){
           for(int j=0;j<a[i].length;j++){
               System.out.print(a[i][j]+" ");
           }
           System.out.println();
       }
       System.out.println();
   }
   
}
