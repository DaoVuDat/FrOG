using FrOG;
using FrOG.Solvers;
using FrOG.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MetaOptims.Solvers
{
    public class AVOA : ISolver
    {
        /// <summary>
        /// Variable vector of final solution.
        /// </summary>
        public double[] Xopt { get; private set; }
        /// <summary>
        /// Cost of final solution.
        /// </summary>
        public double Fxopt { get; private set; }

        //public Dictionary<string, string> settings = new Dictionary<string, string>();

        private readonly Dictionary<string, Dictionary<string, double>> _presets = new Dictionary<string, Dictionary<string, double>>();


        public AVOA()
        {
            //Prepare settings
            var standardSettings = new Dictionary<string, double>
            {
                { "seed", 1},
                { "stepsize", 0.1},
                { "itermax", 1000}
            };

            _presets.Add("AVOA", standardSettings);
        }

        public string GetErrorMessage()
        {
            return "";
        }

        public IEnumerable<string> GetPresetNames()
        {
            return _presets.Keys;
        }

        public bool RunSolver(List<Variable> variables,
            Func<IList<decimal>, double> evaluate,
            string preset,
            string expertsettings,
            string installFolder,
            string documentPath,
            int maxIteration,
            int population)
        {
            var settings = _presets[preset];

            var dvar = variables.Count;
            var lb = new double[dvar];
            var ub = new double[dvar];
            var integer = new bool[dvar];

            for (var i = 0; i < dvar; i++)
            {
                lb[i] = Convert.ToDouble(variables[i].LowerB);
                ub[i] = Convert.ToDouble(variables[i].UpperB);
                integer[i] = variables[i].Integer;
            }

            // Define warppered evaluation function (objective function) from variables to (single double)
            Func<double[], double> eval = x =>
            {
                var decis = x.Select(Convert.ToDecimal).ToList();
                return evaluate(decis);
            };

            try
            {

                var hc = new GWOAlgorithm(lb, ub, population, maxIteration, eval);
                hc.Solve();
                Xopt = hc.get_Xoptimum();
                Fxopt = hc.get_fxoptimum();

                return true;
            }
            catch
            {
                return false;
            }
        }
    }

    internal class GWOAlgorithm
    {
        //source: Stochastic Hill-Climbing, in: Clever Algorithms: Nature-Inspired Programming Recipes (Jason Brownlee)
        //
        //Input: Itermax, ProblemSize 
        //Output: Current 
        //Current  <- RandomSolution(ProblemSize)
        //For (iteri ∈ Itermax )
        //    Candidate  <- RandomNeighbor(Current)
        //    If (Cost(Candidate) >= Cost(Current))
        //        Current  <- Candidate
        //    End
        //End
        //Return (Current)

        /// <summary>
        /// Lower bound for each variable.
        /// </summary>
        public double[] Lb { get; private set; }
        /// <summary>
        /// Upper bound for each variable.
        /// </summary>
        public double[] Ub { get; private set; }
        /// <summary>
        /// Stepsize.
        /// </summary>
        public int Population { get; private set; }
        /// <summary>
        /// Maximum iterations.
        /// </summary>
        public int Itermax { get; private set; }
        /// <summary>
        /// Evaluation function.
        /// </summary>
        public Func<double[], double> Evalfnc { get; private set; }
        /// <summary>
        /// Variable vector of final solution.
        /// </summary>
        public double[] Xopt { get; private set; }
        /// <summary>
        /// Cost of final solution.
        /// </summary>
        public double Fxopt { get; private set; }

        private RandomDistributions rnd;

        /// <summary>
        /// Initialize a stochastic hill climber optimization algorithm. Assuming minimization problems.
        /// </summary>
        /// <param name="lb">Lower bound for each variable.</param>
        /// <param name="ub">Upper bound for each variable.</param>
        /// <param name="itermax">Maximum iterations.</param>
        /// <param name="evalfnc">Evaluation function.</param>
        public GWOAlgorithm(double[] lb, double[] ub, int population, int itermax, Func<double[], double> evalfnc)
        {
            this.Lb = lb;
            this.Ub = ub;
            this.Population = population;
            this.Itermax = itermax;
            this.Evalfnc = evalfnc;

            this.rnd = new RandomDistributions();
        }

        /// <summary>
        /// Minimizes an evaluation function using stochastic hill climbing.
        /// </summary>
        public void Solve()
        {
            int dimension = Lb.Length;

            // Store Best values
            StringBuilder storeBestResults = new StringBuilder();

            List<Wolve> wolves = new List<Wolve>();


            // Alpha, Beta, Delta
            Wolve alphaWolve = new Wolve(dimension, this.Lb, this.Ub);
            alphaWolve.Fitness = double.MaxValue;

            Wolve betaWolve = new Wolve(dimension, this.Lb, this.Ub);
            betaWolve.Fitness = double.MaxValue;

            Wolve deltaWolve = new Wolve(dimension, this.Lb, this.Ub);
            deltaWolve.Fitness = double.MaxValue;


            // Initialization
            for (int i = 0; i < this.Population; i++)
            {
                // Create a new wolve
                Wolve wolve = new Wolve(dimension, this.Lb, this.Ub);

                // Calculate fitness
                double fitness = Evalfnc(wolve.Position);
                wolve.Fitness = fitness;

                wolves.Add(wolve);

                // Find alpha, beta, delta wolve
                if (fitness <= alphaWolve.Fitness)
                {
                    alphaWolve.Position = wolve.CopyPosition();
                    alphaWolve.Fitness = fitness;
                    continue;
                }

                if (fitness <= betaWolve.Fitness)
                {
                    betaWolve.Position = wolve.CopyPosition();
                    betaWolve.Fitness = fitness;
                    continue;
                }

                if (fitness <= deltaWolve.Fitness)
                {
                    deltaWolve.Position = wolve.CopyPosition();
                    deltaWolve.Fitness = fitness;
                }
            }

            // Optimization Loop
            for (int t = 0; t < Itermax; t++)
            {

                double a = 2 - t * (2 / (double)Itermax);

                // Update position of each wolve
                foreach (var wolve in wolves)
                {
                    // Update position of each dimension
                    for (int dim = 0; dim < dimension; dim++)
                    {
                        double pos = wolve.Position[dim];

                        // Update via Alpha
                        double alphaPos = alphaWolve.Position[dim];

                        double r1 = this.rnd.NextDouble();
                        double r2 = this.rnd.NextDouble();

                        double A1 = 2 * a * r1 - a;
                        double C1 = 2 * r2;

                        double DAlpha = Math.Abs(C1 * alphaPos - pos);
                        double X1 = alphaPos - A1 * DAlpha;

                        // Update via Beta
                        double betaPos = betaWolve.Position[dim];

                        r1 = this.rnd.NextDouble();
                        r2 = this.rnd.NextDouble();

                        double A2 = 2 * a * r1 - a;
                        double C2 = 2 * r2;

                        double DBeta = Math.Abs(C2 * betaPos - pos);
                        double X2 = betaPos - A2 * DBeta;

                        // Update via Delta
                        double deltaPos = deltaWolve.Position[dim];

                        r1 = this.rnd.NextDouble();
                        r2 = this.rnd.NextDouble();

                        double A3 = 2 * a * r1 - a;
                        double C3 = 2 * r2;

                        double DDelta = Math.Abs(C3 * deltaPos - pos);
                        double X3 = deltaPos - A3 * DDelta;

                        wolve.Position[dim] = (X1 + X2 + X3) / 3;
                    }
                }

                foreach (var wolve in wolves)
                {

                    for (int dim = 0; dim < dimension; dim++)
                    {
                        double pos = wolve.Position[dim];
                        if (pos > this.Ub[dim])
                        {
                            wolve.Position[dim] = this.Ub[dim];
                        }

                        if (pos < this.Lb[dim])
                        {
                            wolve.Position[dim] = this.Lb[dim];
                        }
                    }

                    // Calculate fitness
                    double fitness = Evalfnc(wolve.Position);
                    wolve.Fitness = fitness;

                    // Find alpha, beta, delta wolve
                    if (fitness <= alphaWolve.Fitness)
                    {
                        alphaWolve.Position = wolve.CopyPosition();
                        alphaWolve.Fitness = fitness;
                        continue;
                    }

                    if (fitness <= betaWolve.Fitness)
                    {
                        betaWolve.Position = wolve.CopyPosition();
                        betaWolve.Fitness = fitness;
                        continue;
                    }

                    if (fitness <= deltaWolve.Fitness)
                    {
                        deltaWolve.Position = wolve.CopyPosition();
                        deltaWolve.Fitness = fitness;
                    }
                }

                this.Xopt = alphaWolve.Position;
                this.Fxopt = (double)alphaWolve.Fitness;

                var roundedPosition = alphaWolve.Position.Select(x => Math.Round(x, 3));

                var positionString = string.Join(";", roundedPosition);
                var fitnessString = Math.Round((double)alphaWolve.Fitness, 3).ToString();

                storeBestResults.AppendLine(positionString + "," + fitnessString);

            }

            try
            {
                File.WriteAllText(@".\results.csv", storeBestResults.ToString());
            }
            catch (Exception ex)
            {
                Console.WriteLine("Data could not be written to the CSV file.");
                return;
            }
        }

        /// <summary>
        /// Get the variable vector of the final solution.
        /// </summary>
        /// <returns>Variable vector.</returns>
        public double[] get_Xoptimum()
        {
            return this.Xopt;
        }

        /// <summary>
        /// Get the cost value of the final solution.
        /// </summary>
        /// <returns>Cost value.</returns>
        public double get_fxoptimum()
        {
            return this.Fxopt;
        }

        private class Wolve
        {
            public double[] Position { get; set; }
            public double? Fitness { get; set; } = null;


            private RandomDistributions rnd;

            public Wolve(int dimension, double[] lb, double[] ub)
            {
                this.rnd = new RandomDistributions();
                this.Position = new double[dimension];
                this.Initialization(dimension, lb, ub);
            }

            private void Initialization(int dimension, double[] lb, double[] ub)
            {
                for (int i = 0; i < dimension; i++)
                {
                    this.Position[i] = rnd.NextDouble() * (ub[i] - lb[i]) + lb[i];
                }
            }

            public double[] CopyPosition()
            {
                return this.Position.ToArray();
            }
        }
    }
}
