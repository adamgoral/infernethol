namespace InferNetHOL.CSharp
{
    using Microsoft.Glee.Drawing;
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    public class NaiveClassifier : ExampleBase
    {
        public NaiveClassifier() : base("Naive Classifier") { }

        public override void Run()
        {
            var inputs = new[] 
            {
                JoinArrays(GetColorAttributeArray(Color.Blue), GetShapeAttributeArray(Shapes.Rectangle), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Red), GetShapeAttributeArray(Shapes.Rectangle), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Blue), GetShapeAttributeArray(Shapes.Star), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Blue), GetShapeAttributeArray(Shapes.Ring), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Green), GetShapeAttributeArray(Shapes.Circle), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Yellow), GetShapeAttributeArray(Shapes.Circle), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Yellow), GetShapeAttributeArray(Shapes.Circle), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Blue), GetShapeAttributeArray(Shapes.Rectangle), new double[] { 15 }),

                JoinArrays(GetColorAttributeArray(Color.Yellow), GetShapeAttributeArray(Shapes.Star), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Red), GetShapeAttributeArray(Shapes.Arrow), new double[] { 10 }),
                JoinArrays(GetColorAttributeArray(Color.Green), GetShapeAttributeArray(Shapes.Trapezium), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Green), GetShapeAttributeArray(Shapes.Diamond), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Yellow), GetShapeAttributeArray(Shapes.Triangle), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Red), GetShapeAttributeArray(Shapes.Ring), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Yellow), GetShapeAttributeArray(Shapes.Circle), new double[] { 15 }),
                JoinArrays(GetColorAttributeArray(Color.Red), GetShapeAttributeArray(Shapes.Ellipse), new double[] { 15 }),
            };

            var outputs = new bool[] { true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false };

            var j = new Range(inputs.Length);
            var noise = Variable.GammaFromMeanAndVariance(1, 1);
            var X = Variable.Observed(inputs.Select(i => Vector.FromArray(i)).ToArray(), j).Named("X");
            var Y = Variable.Observed(outputs, j).Named("Y");
            var weights = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(inputs.First().Length), PositiveDefiniteMatrix.Identity(inputs.First().Length))
                .Named("weights");
            Y[j] = Variable.GaussianFromMeanAndPrecision(Variable.InnerProduct(X[j], weights), noise) > 0;
            var engine = new InferenceEngine();
            var posteriorWeightsDist = engine.Infer<VectorGaussian>(weights);
            var posteriorNoiseDist = engine.Infer<Gamma>(noise);
            weights = Variable.Random(posteriorWeightsDist);
            var testCase = JoinArrays(GetColorAttributeArray(Color.Red), GetShapeAttributeArray(Shapes.Trapezium), new double[] { 15 });
            var testClassification = engine.Infer<Bernoulli>(Variable.InnerProduct(Vector.FromArray(testCase), weights) > 0);
        }

        private static double[] GetColorAttributeArray(Color color)
        {
            return new double[] { color.R, color.G, color.B };
        }

        private static double[] GetShapeAttributeArray(Shapes shape)
        {
            var result = new double[10];
            result[(int)shape] = 1;
            return result;
        }

        private static double[] JoinArrays(params double[][] source)
        {
            var result = new List<double>();
            foreach (var item in source)
            {
                result.AddRange(item);
            }
            
            return result.ToArray();
        }
        
        public enum Shapes
        {
            Rectangle,
            Circle,
            Star,
            Ring,
            Arrow,
            Triangle,
            Ellipse,
            Trapezium,
            Diamond,
            Crescent
        }
    }
}
