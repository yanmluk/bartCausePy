import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
import pandas as pd

from bartCausePy.utils.utils import convert_to_numpy, convert_and_expand

class BARTCause:
    '''Python wrapper for the R bartCause package.'''
    def __init__(self):

        robjects.r('''
        library("bartCause")  
                   
        bart_train <- function(y, Z, X, n_samples,  n_burn,  n_chains) {
                   
          bart_machine <- bartc(y, Z, X, n.samples= n_samples,  n.burn= n_burn,  n.chains= n_chains, seed=99,
                   estimand='ate', method.rsp='bart', method.trt='bart', keepTrees=TRUE, verbose = FALSE)
                
          bart_machine     
        }   

        fit_res <- function(bart_machine, infer_type){
            
          res <- fitted(bart_machine, type=infer_type, sample = c("all"))  
                     
          res
        }
        
        predict_res <- function(bart_machine, new_data, infer_type){
            
          res <- predict(bart_machine, newdata=new_data, group.by=Null, type= infer_type)
          res_lower <- apply(res, 2, quantile, probs = 0.025)
          res_upper <- apply(res, 2, quantile, probs = 0.975)
                   
          #colMeans(res) 
          return(list(y_pred = colMeans(res), lb = res_lower, ub = res_upper))        
        }

        ''')


        self._r_train = robjects.globalenv['bart_train']
        self._r_fit_res =  robjects.globalenv['fit_res']
        self._r_predict_res =  robjects.globalenv['predict_res']

        # enable R to read numpy objects
        rpy2.robjects.numpy2ri.activate()
        # # enable R to read pandas objects
        rpy2.robjects.pandas2ri.activate()


    def fit(self, X, y, Z, n_samples=1000,  n_burn=200,  n_chains=4):
        """ Fit a BART model. 

        Args:
            X: Covariates to fit.
            y: target to predict.
            Z: Treatment column(law)
        """
        # convert to numpy arrays
        X = convert_and_expand(X)
        Z = convert_and_expand(Z)
        try:
            y = y.astype(float)
        except:
            raise TypeError("y must be a continuous array.")
        
        y = convert_to_numpy(y)
        self._bart_machine = self._r_train(y, Z, X, n_samples,  n_burn,  n_chains)
    

    def predict(self, newData, infer_type="ite"):
        """Predict new instance(s)

        Args:
            newData: Covariates(pretreatment variables) or covariates combine with treatment(if infer y or mu).
            infer_type: quantity to infer (options of "mu", "y", "mu.0", "mu.1", "y.0", "y.1", "icate", "ite", "p.score")

        Returns:
            Predicted quantites, credible interval lower bound, credible interval upper bound
        Note:
            ite infers effect between of observed and estimated counterfactual while icate infers estimated factual and counterfactual.
        """
        if infer_type in ["y", "mu", "mu.1", "mu.0", "y.1", "y.0"]:
            dfNewData = pd.DataFrame(newData, columns=([('V'+str(i)) for i in range(1,newData.shape[1])] + ['z']))
            res = self._r_predict_res(self._bart_machine, dfNewData, infer_type)
            return res[0], res[1], res[2]
        else:
            res = self._r_predict_res(self._bart_machine, newData, infer_type)
            return res[0], res[1], res[2]


    def fitted(self, infer_type="ite"):
        """Get fitted values

        Args:
            infer_type: quantity to infer (options of "pate", "sate", "cate", "mu.obs", "mu.cf", "mu.0",
            "mu.1", "y.cf", "y.0", "y.1", "icate", "ite",
            "p.score", "p.weights")

        Returns:
            fitted quantites.
        """
        return self._r_fit_res(self._bart_machine, infer_type)

