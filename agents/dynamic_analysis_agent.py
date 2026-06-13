import logging
import pandas as pd
from typing import Optional, Dict, Any
import numpy as np
from models import MLPClassifier, MLPRegressor, CVAEGenerator, DiffusionGenerator
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import get_tool_decider, create_data_summary, ToolDecider

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class DynamicAnalysisAgent:
    """
    Rule-based SLM for dynamic analysis tool selection.
    Chooses between multiple classifiers and anomaly detection based on data/task.
    """
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None, task: str = "classification", 
                 params: Dict[str, Any] = None, tool_decider: Optional[ToolDecider] = None):
        self.data = data
        self.target_column = target_column
        self.task = task
        self.params = params or {}
        self.tool_decider = tool_decider or get_tool_decider("rule_based")
        self.model = None
        self.model_name = None
        self.results = {}
        self.tried_models = []  # Track models already tried
        self.best_performance = -float('inf')  # Track best performance
        self.best_model = None
        self.best_results = None
        logging.info(f"DynamicAnalysisAgent initialized for task: {task} with params: {self.params}")

    def choose_tool(self) -> str:
        """
        LLM-based selection of analysis tool using ToolDecider.
        Returns the name of the chosen tool.
        """
        if self.task == "regression":
            self.model_name = "MLPRegressor"
            logging.info(f"Regression task, selected tool: {self.model_name}")
        else:
            # Use ToolDecider for model selection
            data_summary = create_data_summary(self.data)
            if self.task == "classification":
                available_models = "MLPClassifier"
            elif self.task == "generation":
                available_models = ["CVAEGenerator", "DiffusionGenerator"]
            else:
                available_models = "MLPClassifier"  # Default to classification model for unknown tasks
            
            decision = self.tool_decider.decide_model_family(self.task, data_summary, available_models)
            default_model = "MLPRegressor" if self.task == "regression" else "MLPClassifier"
            self.model_name = decision.get("model", default_model)
            logging.info(f"ToolDecider selected tool: {self.model_name}, reason: {decision.get('reason', 'N/A')}")
            
        return self.model_name

    def run(self, force_retry: bool = False) -> Dict[str, Any]:
        """
        Executes the chosen analysis tool and returns results.
        If performance is poor and force_retry=True, try multiple models.
        """
        if self.task == "generation":
            return self._run_generation()
            
        if self.target_column is None:
            logging.error("Target column required learning tasks.")
            return None
        
        # If force_retry is True, try multiple models to find the best one
        if force_retry:
            return self._try_multiple_models()
            
        tool = self.choose_tool()
        if tool == "MLPRegressor":
            return self._run_mlp_regressor()
        elif tool == "MLPClassifier":
            return self._run_mlp_classifier()
     
        else:
            if self.task == "regression":
                return self._run_mlp_regressor()
            return self._run_mlp_classifier()

    # def _try_multiple_models(self) -> Optional[Dict[str, Any]]:
    #     """
    #     Try multiple models and return the best performing one.
    #     """
    #     logging.info("🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...")
        
    #     # Define model candidates based on task type
    #     if self.task == "classification":
    #         model_candidates = [
    #             ("RandomForestClassifier", self._run_random_forest),
    #             ("LogisticRegression", self._run_logistic_regression),
    #             ("SVC", self._run_svc)
    #         ]
    #     elif self.task == "regression":
    #         model_candidates = [
    #             ("LinearRegression", self._run_linear_regression),
    #             ("Ridge", self._run_ridge),
    #             ("Lasso", self._run_lasso),
    #             ("RandomForestRegressor", self._run_random_forest_regressor),
    #             ("SVR", self._run_svr)
    #         ]
    #     else:
    #         return None
        
    #     best_performance = -float('inf')
    #     best_model_name = None
    #     best_results = None
        
    #     for model_name, model_func in model_candidates:
    #         if model_name in self.tried_models:
    #             logging.info(f"⏭️ Skipping {model_name} (already tried)")
    #             continue
                
    #         try:
    #             logging.info(f"🔄 Trying {model_name}...")
    #             results = model_func()
                
    #             if results:
    #                 # Calculate performance metric
    #                 if self.task == "classification":
    #                     performance = results.get("accuracy", 0)
    #                 else:  # regression
    #                     performance = results.get("r2", -float('inf'))
                    
    #                 logging.info(f"   {model_name} performance: {performance:.4f}")
                    
    #                 if performance > best_performance:
    #                     best_performance = performance
    #                     best_model_name = model_name
    #                     best_results = results
    #                     self.best_model = model_name
    #                     self.best_results = results
    #                     self.best_performance = performance
                
    #             self.tried_models.append(model_name)
                
    #         except Exception as e:
    #             logging.warning(f"   {model_name} failed: {str(e)}")
    #             self.tried_models.append(model_name)
    #             continue
        
    #     if best_results and best_performance > -float('inf'):
    #         logging.info(f"🏆 Best model: {best_model_name} (performance: {best_performance:.4f})")
    #         return best_results
    #     else:
    #         logging.error("❌ All models failed or produced invalid results")
    #         return None
    # I don't have multiple models, I have one model for each task type, expect for generation where I have 2. 

    def _run_mlp_classifier(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = MLPClassifier()
        self.model.train(X_train, y_train)
        preds = self.model.test(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"MLP Classifier accuracy: {acc:.4f}")
        return {
            "model": "MLPClassifier",
            "accuracy": acc,
            "classification_report": report,
            "predictions": preds,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }

    def _run_mlp_regressor(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = MLPRegressor()
        self.model.train(X_train, y_train)
        preds = self.model.test(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"MLP Regressor accuracy: {acc:.4f}")
        return {
            "model": "MLPRegressor",
            "accuracy": acc,
            "classification_report": report,
            "predictions": preds,
            "X_test": X_test,
            "y_test": y_test
        }

    def _run_diffiusion_generator(self) -> Dict[str, Any]:
        # Drop target and any ID columns (pass-through identifiers)
        X = self.data.drop(columns=[self.target_column])
        id_columns = [col for col in X.columns if 'ID' in col.upper()]
        if id_columns:
            logging.info(f"Dropping ID columns from model training: {id_columns}")
            X = X.drop(columns=id_columns)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = DiffusionGenerator()
        self.model.train(X_train, y_train)
        preds = self.model.test(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        logging.info(f"Diffusion Generator accuracy: {acc:.4f}")
        return {
            "model": "DiffusionGenerator",
            "n_ok_samples": self.aemodel.n_ok_samples,
            "n_bad_samples": self.aemodel.n_bad_samples,
            "predictions": preds,
            "X_test": X_test,
            "y_test": y_test
        }


if __name__ == "__main__":
    # Example usage
    logging.info("--- Running DynamicAnalysisAgent in Standalone Mode ---")
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1,2,3,4,5,6,7,8,9,10],
        'B': [2,3,4,5,6,7,8,9,10,11],
        'target': [0,1,0,1,0,1,0,1,0,1]
    })
    agent = DynamicAnalysisAgent(df, target_column='target')
    results = agent.run()
    logging.info(f"Results: {results}")
