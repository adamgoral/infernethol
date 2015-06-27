namespace InferNetHOL.Clustering.Models
{
    using InferNetHOL.Clustering.Services;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using System.Windows.Input;

    public class GaussianMixtureViewModel : INotifyPropertyChanged
    {
        private readonly GaussianMixtureModel gaussianMixtureModel;

        private readonly DataSource dataSource;

        private string selectedDataSetName;

        private InferenceResultItem selectedInferenceResult;

        public GaussianMixtureViewModel(DataSource dataSource)
        {
            this.dataSource = dataSource;
            this.gaussianMixtureModel = new GaussianMixtureModel();
            this.InferCommand = new DelegateCommand<string>(selected => this.ExecuteInferCommand(selected), _ => true);
            this.DataSetNames = dataSource.GetDataSetsNames().ToList();
            this.SelectedDataSetName = this.DataSetNames.FirstOrDefault();
        }

        public ICommand InferCommand { get; private set; }

        public List<string> DataSetNames { get; private set; }

        public List<InferenceResultItem> InferenceResults { get; private set; }

        public List<DatasetColumn> SourceColumns { get; private set; }

        private string chartRowField;
        private string chartColumnField;

        public string ChartRowField
        {
            get { return this.chartRowField; }
            set
            {
                if (this.chartRowField != value)
                {
                    this.chartRowField = value;
                    this.OnPropertyChanged("ChartRowField");
                    DisplayPlot();
                }
            }
        }

        public string ChartColumnField 
        {
            get { return this.chartColumnField; }
            set
            {
                if (this.chartColumnField != value)
                {
                    this.chartColumnField = value;
                    this.OnPropertyChanged("ChartColumnField");
                    DisplayPlot();
                }
            }
        }

        public List<string> ChartFields { get; private set; }

        private void DisplayPlot()
        {
            var rowIndex = this.ChartFields.IndexOf(this.chartRowField);
            var columnIndex = this.ChartFields.IndexOf(this.chartColumnField);
            if (rowIndex == -1 || columnIndex == -1)
                return;
            var series = new LabelledSeries<Tuple<double, double>>("data", data.Select(d => Tuple.Create(d[rowIndex], d[columnIndex]))) { IsScatter = true };
            this.Series = new[] {series};
            this.OnPropertyChanged("Series");
        }

        private Vector[] data;

        public string SelectedDataSetName
        {
            get { return this.selectedDataSetName; }
            set 
            {
                if (this.selectedDataSetName != value)
                {
                    this.selectedDataSetName = value;
                    this.OnPropertyChanged("SelectedDataSetName");
                    this.SourceColumns = this.dataSource.GetColumns(value).Select(c => new DatasetColumn(c)).ToList();
                    this.OnPropertyChanged("SourceColumns");
                    this.ChartFields = this.SourceColumns.Select(c => c.Name).ToList();
                    this.OnPropertyChanged("ChartFields");
                    this.data = this.dataSource.Load(this.selectedDataSetName).ToArray();
                    this.ChartRowField = this.ChartFields.FirstOrDefault();
                    this.ChartColumnField = this.ChartFields.LastOrDefault();
                }
            }
        }

        public LabelledSeries<Tuple<double, double>>[] Series { get; private set; }

        public string SelectedInferenceResultText { get; private set; }

        public InferenceResultItem SelectedInferenceResult
        {
            get { return this.selectedInferenceResult; }
            set
            {
                if (this.selectedInferenceResult != value)
                {
                    this.selectedInferenceResult = value;
                    this.OnPropertyChanged("SelectedInferenceResult");
                    this.SelectedInferenceResultText = ToString(value);
                    this.OnPropertyChanged("SelectedInferenceResultText");
                }
            }
        }

        private static string ToString(InferenceResultItem item)
        {
            if (item == null)
                return null;

            return string.Join(Environment.NewLine, item.Clusters.Select(ToString));
        }

        private static string ToString(Cluster cluster)
        {
            return string.Format("Means: {0}\nCovariance:\n{1}", ToString(cluster.Means), ToString(cluster.Covariance));
        }

        private static string ToString(Vector vector)
        {
            return string.Join(",", vector.Select(item => Math.Round(item, 2)));
        }

        private static string ToString(PositiveDefiniteMatrix matrix)
        {
            var sb = new StringBuilder();
            for (var r = 0; r < matrix.Rows; r++)
            {
                var sb2 = new StringBuilder();
                for (var c = 0; c < matrix.Cols; c++)
                {
                    sb2.Append(matrix[r, c]);
                    sb2.Append("\t");
                }

                sb.AppendLine(sb2.ToString());
            }

            return sb.ToString();
        }

        public void ExecuteInferCommand(string dataSetName)
        {
            var selectedColumnIndices = GetSelectedColumnIndices(this.SourceColumns);
            var source = this.dataSource.Load(dataSetName).ToArray();
            var data = source.Select(v => Vector.FromArray(selectedColumnIndices.Select(i => v[i]).ToArray())).ToArray();
            var results = Enumerable.Range(1, 9)
                .AsParallel()
                .AsOrdered()
                .Select(i => this.gaussianMixtureModel.InferStandard(data, i))
                .ToArray();
            var bpm = new BayesPointMachineModel();
            var input = source.Select(v => v[0]).ToArray();
            var bpmResults = Enumerable.Range(1, 9)
                .AsParallel()
                .AsOrdered()
                .Select(i => bpm.Regression(data, input, i))
                .ToArray();
            var estimated = bpmResults.Select(r => Tuple.Create(input, r.Result.Select(c => Estimate(c.Weights, data).ToArray()).ToArray())).ToArray();
            this.InferenceResults = results.Select(r => new InferenceResultItem(r)).ToList();
            this.OnPropertyChanged("InferenceResults");
        }

        public static IEnumerable<double> Estimate(Vector weights, Vector[] data)
        {
            foreach(var item in data)
            {
                yield return Vector.InnerProduct(weights, item);
            }
        }

        private static IEnumerable<int> GetSelectedColumnIndices(List<DatasetColumn> list)
        {
            for (var i = 0; i < list.Count; i++)
            {
                if (list[i].Selected)
                {
                    yield return i;
                }
            }
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

        public event PropertyChangedEventHandler PropertyChanged;

        private void OnPropertyChanged(string propertyName)
        {
            var handler = this.PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }
}
