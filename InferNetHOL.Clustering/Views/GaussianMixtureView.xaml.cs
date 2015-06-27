using InferNetHOL.Clustering.Models;
using InferNetHOL.Clustering.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace InferNetHOL.Clustering.Views
{
    /// <summary>
    /// Interaction logic for GaussianMixtureView.xaml
    /// </summary>
    public partial class GaussianMixtureView : UserControl
    {
        public GaussianMixtureView()
        {
            InitializeComponent();
            var currentPath = new Uri("file:///" + Environment.CurrentDirectory);
            var dataPath = new Uri(currentPath, Properties.Settings.Default.dataPath);
            this.ViewModel = new GaussianMixtureViewModel(new DataSource(dataPath));
        }

        public GaussianMixtureViewModel ViewModel
        {
            get { return this.DataContext as GaussianMixtureViewModel; }
            set { this.DataContext = value; }
        }
    }
}
