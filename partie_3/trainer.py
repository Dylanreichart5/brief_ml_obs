from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        
    def run(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data['X'], self.data['y'], test_size=0.2, random_state=42)
        self.pipeline.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        score = self.pipeline.score(self.X_test, self.y_test)
        return score
