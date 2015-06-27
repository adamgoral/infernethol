namespace InferNetHOL.Clustering.Views
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows;
    using System.Windows.Controls;
    using System.Windows.Data;
    using System.Windows.Media;

    using Models;
    using OxyPlot;
    using OxyPlot.Axes;
    using OxyPlot.Series;

    /// <summary>
    /// Interaction logic for GraphView.xaml
    /// </summary>
    public partial class GraphView : UserControl
    {
        public GraphView()
        {
            InitializeComponent();
            this.DataContextChanged += GraphView_DataContextChanged;
        }

        void GraphView_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            // TODO: Temporary measure until proper fix for sub item property binding is found
            //var source = e.NewValue as IExample;
            //if (source != null)
            //{
            //    var binding = new Binding("Series");
            //    this.SetBinding(SeriesProperty, binding);
            //}
        }

        public static readonly DependencyProperty SeriesProperty = DependencyProperty.Register("Series", typeof(LabelledSeries<Tuple<double, double>>[]), typeof(GraphView), new PropertyMetadata(new PropertyChangedCallback(SeriesChangedCallback)));

        public LabelledSeries<Tuple<double, double>>[] Series
        {
            get { return (LabelledSeries<Tuple<double, double>>[])GetValue(SeriesProperty); }
            set
            {
                SetValue(SeriesProperty, value);
            }
        }

        private static void SeriesChangedCallback(DependencyObject target, DependencyPropertyChangedEventArgs args)
        {
            var view = target as GraphView;
            if (view == null)
                return;
            var data = args.NewValue as LabelledSeries<Tuple<double, double>>[];
            view.Display(data);
        }

        private void Display(LabelledSeries<Tuple<double, double>>[] dataSeries)
        {
            var plotModel = new PlotModel();
            var xAxis = new LinearAxis
            {
                Position = AxisPosition.Bottom
            };

            plotModel.Axes.Add(xAxis);
            plotModel.Axes.Add(new LinearAxis());
            plotModel.LegendBorder = OxyColors.Black;
            plotModel.LegendBorderThickness = 1;

            if (dataSeries == null)
                return;
            var LineColors = new[] { Colors.Blue, Colors.Red, Colors.Violet, Colors.Brown, Colors.Green, Colors.Gold, Colors.Olive };

            for (var i = 0; i < dataSeries.Length; i++)
            {
                var timeSeriesItem = dataSeries[i];
                var colorIndex = i % (LineColors.Length - 1);
                var lineColor = LineColors[colorIndex];
                var count = timeSeriesItem.Series.Length;
                if (count > 0)
                {
                    var series = new ScatterSeries();
                    series.Title = timeSeriesItem.Label;
                    series.DataFieldX = "Item1";
                    series.DataFieldY = "Item2";
                    series.MarkerType = MarkerType.Circle;
                    series.MarkerSize = 3;
                    series.MarkerFill = OxyColor.FromArgb(lineColor.A, lineColor.R, lineColor.G, lineColor.B);
                    series.ItemsSource = timeSeriesItem.Series;
                    plotModel.Series.Add(series);
                }
            }

            plotter.Model = plotModel;
        }
    }
}
