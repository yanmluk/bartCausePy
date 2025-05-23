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
                   
        inclusion_prop <- function(bart_machine, n_samples){
          bart_response_fit <- bart_machine$fit.rsp
                   
          # Extract variable inclusion counts
          varcount_matrix <- bart_response_fit$varcount
                   
          # Sum varcounts(inclusion counts) across the first two dimensions (chains and samples)
          total_varcount <- apply(varcount_matrix, 3, sum)
        
          # Compute the percentage of use for each variable(inclusion counts/total number of splits)
          percent_vars <- varcount_matrix / total_varcount
          percent_vars <- percent_vars *n_samples
                   
          # Collapse the first two dimensions (chains and samples) to compute mean percentage
          # Use 'aperm' to adjust dimensions for 'apply'
          collapsed_percent_vars <- aperm(percent_vars, c(3, 1, 2))
          res <- apply(collapsed_percent_vars, 1, mean)
          # remove non-covariates
          res$z <- NULL
          res$ps <- NULL
                        
          res
        }

        ''')


        self._r_train = robjects.globalenv['bart_train']
        self._r_fit_res =  robjects.globalenv['fit_res']
        self._r_predict_res =  robjects.globalenv['predict_res']
        self._r_inclusion_prop =  robjects.globalenv['inclusion_prop']

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
        if isinstance(X, pd.core.frame.DataFrame):
            pass
        else:
            # convert to numpy arrays
            X = convert_and_expand(X)
        if isinstance(Z, pd.core.frame.DataFrame): 
            pass
        else: 
            # convert to numpy arrays
            Z = convert_and_expand(Z)
        
        y = convert_to_numpy(y)
        self._bart_machine = self._r_train(y, Z, X, n_samples,  n_burn,  n_chains)
        self._n_samples = n_samples
    

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
        assert hasattr(self, "_bart_machine"), "Did not fit the BART model"

        if infer_type in ["y", "mu", "mu.1", "mu.0", "y.1", "y.0"]:
            if isinstance(newData, pd.core.frame.DataFrame):
                dfNewData = newData.copy(deep=True)
                dfNewData.rename(columns={newData.columns[-1]:'z'}, inplace=True)
            else:
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
        assert hasattr(self, "_bart_machine"), "Did not fit the BART model"
        return self._r_fit_res(self._bart_machine, infer_type)
    

    def get_feature_importance(self):
        """Get feature importance of covariates

        Returns:
            List of covariates' importance
        """
        assert hasattr(self, "_bart_machine"), "Did not fit the BART model"
        assert hasattr(self, "_n_samples"), "BART has no attribute number of samples"
        feature_importance_obj = self._r_inclusion_prop(self._bart_machine, self._n_samples)
        names = feature_importance_obj.names
        feature_importance_dict = {}
        for name in names:
            feature_importance_dict[name] = feature_importance_obj.rx2(str(name))[0]

        return feature_importance_dict


