import sys
import requests
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

class BitcoinApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Prediction Maker")
        self.setStyleSheet("background-color: #111; color: white; font-size: 14px;")
        layout = QVBoxLayout()

        self.btn_live = QPushButton("💰 Fetch Live BTC Price")
        self.btn_live.clicked.connect(self.fetch_live_price)
        layout.addWidget(self.btn_live)

        self.btn_chart = QPushButton("📉 Show Historical Chart")
        self.btn_chart.clicked.connect(self.show_chart)
        layout.addWidget(self.btn_chart)

        self.btn_predict = QPushButton("🛡️ Predict Next 10-min Price")
        self.btn_predict.clicked.connect(self.predict_price)
        layout.addWidget(self.btn_predict)

        for btn in [self.btn_live, self.btn_chart, self.btn_predict]:
            btn.setStyleSheet("background-color: #222; padding: 10px;")

        self.setLayout(layout)

    def fetch_live_price(self):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(url)
            price = response.json()["bitcoin"]["usd"]
            QMessageBox.information(self, "Live BTC Price", f"₿ Current BTC Price: ${price:,.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch live price.\n{str(e)}")

    def get_binance_data(self):
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "_", "_", "_", "_", "_"
        ])
        df["close"] = df["close"].astype(float)
        return df

    def show_chart(self):
        try:
            df = self.get_binance_data()
            plt.style.use("dark_background")
            plt.plot(df["close"], label="Close Price", color="cyan")
            plt.title("Bitcoin - Last 60 Min (Binance)")
            plt.xlabel("Minute")
            plt.ylabel("Price (USDT)")
            plt.legend()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load chart.\n{str(e)}")

    def predict_price(self):
        try:
            df = self.get_binance_data()
            df["minute"] = list(range(len(df)))
            model = LinearRegression()
            model.fit(df[["minute"]], df["close"])
            next_price = model.predict([[df["minute"].max() + 10]])[0]
            QMessageBox.information(
                self, "Prediction", f"📈 Predicted BTC Price in 10 min: ${next_price:,.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed.\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BitcoinApp()
    window.resize(400, 200)
    window.show()
    sys.exit(app.exec_())
