package br.com.inteligenciaartificial.algoritmos.neural;

public class MathUtils {
	public static double sigmoid(final double z) {
		return 1 / (1 + Math.exp(-z));
	}

	public static double sigmoidDerivative(final double z) {
		final double sig = MathUtils.sigmoid(z);
		return sig * (1 - sig);
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
