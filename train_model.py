from model import ChatbotModel

def train():
    print("Training model...")
    model = ChatbotModel()
    results = model.train('data/dataset.csv')
    print(f"✅ Accuracy: {results['accuracy']:.0%}")
    print("✅ Model saved")

if __name__ == '__main__':
    train()