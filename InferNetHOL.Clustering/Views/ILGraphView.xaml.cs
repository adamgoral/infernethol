namespace InferNetHOL.Clustering.Views
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows;
    using System.Windows.Controls;
    using System.Windows.Data;
    using System.Drawing;

    using Models;
    using ILNumerics.Drawing;
    using ILNumerics.Drawing.Plotting;
    using ILNumerics;

    /// <summary>
    /// Interaction logic for GraphView.xaml
    /// </summary>
    public partial class ILGraphView : UserControl
    {
        public ILGraphView()
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

        public static readonly DependencyProperty SeriesProperty = DependencyProperty.Register("Series", typeof(LabelledSeries<Tuple<double, double>>[]), typeof(ILGraphView), new PropertyMetadata(new PropertyChangedCallback(SeriesChangedCallback)));

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
            var view = target as ILGraphView;
            if (view == null)
                return;
            var data = args.NewValue as LabelledSeries<Tuple<double, double>>[];
            view.Display(data);
        }

        private void Display(LabelledSeries<Tuple<double, double>>[] dataSeries)
        {
            var scene = new ILScene();

            if (dataSeries == null)
                return;
            var PlotColors = new[] { Color.Blue, Color.Red, Color.Violet, Color.Brown, Color.Green, Color.Gold, Color.Olive };

            for (var i = 0; i < dataSeries.Length; i++)
            {
                var timeSeriesItem = dataSeries[i];
                var colorIndex = i % (PlotColors.Length - 1);
                var plotColor = PlotColors[colorIndex];
                var count = timeSeriesItem.Series.Length;
                if (count > 0)
                {
                    var plotCube = new ILPlotCube();
                    ILArray<float> array = GetMatrix(new[] { timeSeriesItem.Series.Select(s => (float)s.Item1).ToArray(), timeSeriesItem.Series.Select(s => (float)s.Item2).ToArray(), timeSeriesItem.Series.Select(s => (float)0).ToArray() });
                    plotCube.Add(new ILPoints
                    {
                        Positions = array,
                        Color = plotColor
                    });

                    scene.Add(plotCube);
                }
            }

            plotter.Scene = scene;
            plotter.Refresh();
        }

        static float[,] GetMatrix(float[][] source)
        {
            var result = new float[source[0].Length, source.Length];
            for (var x = 0; x < result.GetLength(0); x++)
            {
                for (var y = 0; y < result.GetLength(1); y++)
                {
                    result[x, y] = source[y][x];
                }
            }

            return result;
        }
    }
}
