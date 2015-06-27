using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferNetHOL.CSharp
{
    public class GaussianMixture : ExampleBase
    {
        public GaussianMixture()
            : base("Gaussian Mixture")
        {
        }

        public override void Run()
        {
            var clusters = 3;
            var dimensions = 2;
            var data = CreateTestData(dimensions, clusters).Take(300).ToArray();
            var results = new List<double>();
            for (var i = 2; i < 6; i++)
            {
                results.Add(InferMixture(data, dimensions, i));
            }
        }

        private static double InferMixture(Vector[] observedData, int dimensions, int clusters)
        {
            var evidence = Variable.Bernoulli(0.5).Named("evidence");

            var evidenceBlock = Variable.If(evidence);
            var k = new Range(clusters).Named("k");

            // Mixture component means
            var means = Variable.Array<Vector>(k).Named("means");
            means[k] = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Zero(dimensions),
                PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01)).ForEach(k);

            // Mixture component precisions
            var precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
            precs[k] = Variable.WishartFromShapeAndRate(100.0, PositiveDefiniteMatrix.IdentityScaledBy(dimensions, 0.01)).ForEach(k);

            // Mixture weights 
            var weights = Variable.Dirichlet(k, Enumerable.Range(0, clusters).Select(_ => 1.0).ToArray()).Named("weights");

            // Create a variable array which will hold the data
            var n = new Range(observedData.Length).Named("n");
            var data = Variable.Array<Vector>(n).Named("x");

            // Create latent indicator variable for each data point
            var z = Variable.Array<int>(n).Named("z");

            // The mixture of Gaussians model
            using (Variable.ForEach(n))
            {
                z[n] = Variable.Discrete(weights);
                using (Variable.Switch(z[n]))
                {
                    data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]]);
                }
            }

            // Initialise messages randomly so as to break symmetry
            var zinit = new Discrete[n.SizeAsInt];
            for (int i = 0; i < zinit.Length; i++)
                zinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            z.InitialiseTo(Distribution<int>.Array(zinit));

            evidenceBlock.CloseBlock();

            // Attach some generated data
            data.ObservedValue = observedData.ToArray();

            // The inference
            var ie = new InferenceEngine(new VariationalMessagePassing());
            ie.ShowProgress = false;
            Console.WriteLine("Dist over pi=" + ie.Infer(weights));
            Console.WriteLine("Dist over means=\n" + ie.Infer(means));
            Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));
            var logEvidence = ie.Infer<Bernoulli>(evidence);
            Console.WriteLine("The model log evidence is {0}", logEvidence.LogOdds);
            return logEvidence.LogOdds;
        }

        private static IEnumerable<Vector> CreateTestData(int dimensions, int clusters)
        {
            var dimRange = new Range(dimensions);
            var clusterRange = new Range(clusters);
            var trueMeans = Variable.Array<Vector>(clusterRange);
            var trueCovariance = Variable.Array<PositiveDefiniteMatrix>(clusterRange);
            var trueVectorGaussian = new VectorGaussian[clusters];
            for (var i = 0; i < clusters; i++)
            {
                var trueMeansVector = Vector.FromArray(CreateRandomArray(dimensions));
                Console.WriteLine("True means {0}", trueMeansVector);
                trueVectorGaussian[i] = VectorGaussian.FromMeanAndPrecision(trueMeansVector, PositiveDefiniteMatrix.Identity(dimensions));
            }

            var vectorGaussians = Variable.Array<VectorGaussian>(clusterRange);
            var dirichletMixture = Dirichlet.Uniform(clusters).Sample();
            Console.WriteLine("True mixture {0}", dirichletMixture);

            while (true)
            {
                var index = Rand.Sample(dirichletMixture);
                yield return trueVectorGaussian[index].Sample();
            }
        }

        private static Random rand = new Random(12347);

        private static void ResetRand()
        {
            rand = new Random(12347);
        }

        private static double[] CreateRandomArray(int size)
        {
            return Enumerable.Range(0, size).Select(_ => rand.NextDouble() * 100).ToArray();
        }
    }
}
