package br.com.inteligenciaartificial.algoritmos.neural;

public class MathUtils {
	public static double sigmoid(final double z) {
		return 1d / (1d + Math.exp(-z));
	}

	public static double sigmoidDerivative(final double z) {
		final double sig = MathUtils.sigmoid(z);
		// Essa eh a expressao algebrica da derivada da funcao sigmoid.
		return sig * (1 - sig);
	}

	public static double step(final double z) {
		return z > 0 ? 1 : -1;
	}

	public static double tanh(final double x) {
		return Math.tanh(x);
	}

	public static double tanhDerivative(final double z) {
		final double tan = MathUtils.tanh(z);
		return 1 - tan * tan;
	}

	private MathUtils() {
	}
}
