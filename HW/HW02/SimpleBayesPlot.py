import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class SimpleBayesPlot:
    def __init__(self, mu1, var1, mu2, var2, prior_w1=0.5, prior_w2=0.5):
        self.mu1 = mu1
        self.var1 = var1
        self.sd1 = var1**0.5
        self.mu2 = mu2
        self.var2 = var2
        self.sd2 = var2**0.5
        self.prior_w1 = prior_w1
        self.prior_w2 = prior_w2
        self.decision_boundary = None

    def calculate_posteriors(self, x_values):
        likelihood_w1 = norm.pdf(x_values, loc=self.mu1, scale=self.sd1)
        likelihood_w2 = norm.pdf(x_values, loc=self.mu2, scale=self.sd2)

        posterior_w1 = likelihood_w1 * self.prior_w1
        posterior_w2 = likelihood_w2 * self.prior_w2

        return posterior_w1, posterior_w2

    def find_decision_boundary(self, x_values):
        self.decision_boundary = x_values[np.argmin(np.abs(self.calculate_posteriors(x_values)[0] - self.calculate_posteriors(x_values)[1]))]

    def plot_posteriors(self, x_values):
        if self.decision_boundary is None:
            self.find_decision_boundary(x_values)

        posterior_w1, posterior_w2 = self.calculate_posteriors(x_values)

        # Plot posteriors
        plt.plot(x_values, posterior_w1, label='Posterior for Happy Cat (w1)')
        plt.plot(x_values, posterior_w2, label='Posterior for Sad Cat (w2)')

        # Plot decision boundary
        plt.axvline(x=self.decision_boundary, color='r', linestyle='--', label='Decision Boundary')
        
        # Add text annotation for decision boundary value
        plt.text(self.decision_boundary, max(max(posterior_w1), max(posterior_w2)),
             f'Decision Boundary (x = {round(self.decision_boundary, 2)})',
             verticalalignment='bottom', horizontalalignment='right', color='r')

        # Add labels and legend
        plt.xlabel('Normalized Amount of Food Eaten by Cat (x)')
        plt.ylabel('Posterior Probability')
        plt.title('Posterior Probabilities for Cat Emotions')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Generate standard normalized x values
    x_values = np.linspace(-3, 3, 1000)

    # Create the classifier instance
    classifier = SimpleBayesPlot(mu1=4, var1=2, mu2=0, var2=2)

    # Plot posteriors graph
    classifier.plot_posteriors(x_values)

    print(f'Decision Boundary: x = {classifier.decision_boundary}')