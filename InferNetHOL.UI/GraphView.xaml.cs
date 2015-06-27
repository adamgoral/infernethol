namespace InferNetHOL.UI
{
    using InferNetHOL.CSharp;
    using Microsoft.Research.DynamicDataDisplay;
    using Microsoft.Research.DynamicDataDisplay.Charts;
    using Microsoft.Research.DynamicDataDisplay.DataSources;
    using Microsoft.Research.DynamicDataDisplay.PointMarkers;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows;
    using System.Windows.Controls;
    using System.Windows.Data;
    using System.Windows.Media;

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
            var source = e.NewValue as IExample;
            if (source != null)
            {
                var binding = new Binding("Series");
                this.SetBinding(SeriesProperty, binding);
            }
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

        private void Display(LabelledSeries<Tuple<double, double>>[] timeSeries)
        {
            plotter.RemoveUserElements();
            plotter.Legend.LegendRight = double.NaN;
            plotter.Legend.LegendLeft = 10;

            if (timeSeries == null)
                return;
            var LineColors = new[] { Colors.Blue, Colors.Red, Colors.Violet, Colors.Brown, Colors.Green, Colors.Gold, Colors.Olive };

            for (var i = 0; i < timeSeries.Length; i++)
            {
                var timeSeriesItem = timeSeries[i];
                var colorIndex = i % (LineColors.Length - 1);
                var lineColor = LineColors[colorIndex];
                var count = timeSeriesItem.Series.Length;
                if (count > 0)
                {
                    var xSource = new EnumerableDataSource<double>(timeSeriesItem.Series.Select(s => s.Item1));
                    xSource.SetXMapping(x => x);
                    var ySource = new EnumerableDataSource<double>(timeSeriesItem.Series.Select(s => s.Item2));
                    ySource.SetYMapping(Convert.ToDouble);
                    var compSource = new CompositeDataSource(xSource, ySource);
                    if (timeSeriesItem.IsScatter)
                    {
                        var pen = new Pen(new SolidColorBrush(Colors.Transparent), 0);
                        var marker = new CircleElementPointMarker();
                        var description = new PenDescription(timeSeriesItem.Label);
                        plotter.AddLineGraph(compSource, pen, marker, description);
                    }
                    else
                    {
                        plotter.AddLineGraph(compSource, lineColor, 1, timeSeriesItem.Label);
                    }
                }
            }

            plotter.Viewport.FitToView();
        }
    }
}
