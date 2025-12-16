import tensorflow as tf

class BinaryF1(tf.keras.metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.p = tf.keras.metrics.Precision()
        self.r = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.p.update_state(y_true, y_pred)
        self.r.update_state(y_true, y_pred)

    def result(self):
        return 2 * self.p.result() * self.r.result() / (
            self.p.result() + self.r.result() + 1e-7
        )

    def reset_state(self):
        self.p.reset_state()
        self.r.reset_state()

