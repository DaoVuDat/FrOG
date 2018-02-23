﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace FrOG
{
    internal static class OptimizationLoop
    {
        //Settings
        public static bool BolMaximize;
        public static bool BolMaxIter;
        public static int MaxIter;
        public static bool BolMaxIterNoProgress;
        public static int MaxIterNoProgress;
        public static bool BolMaxDuration;
        public static double MaxDuration;
        public static bool BolRuns;
        public static int Runs;
        public static int PresetIndex;
        public static bool BolRandomize = true;

        //Expertsettings for RBFOpt
        public static string ExpertSettings;

        //BolLog Settings
        public static bool BolLog;
        public static string LogName;

        //Variables
        private static BackgroundWorker _worker;
        public static OptimizationComponent _component; //do not change name -- suggestion Component overlaps with System.ComponentModel.Component

        private static int _iterations;
        private static int _iterationsCurrentBest;
        private static double _bestValue;
        private static IList<decimal> _bestParams;
        private static OptimizationResult.ResultType _resultType;

        private static Log _log;
        private static Stopwatch _stopwatchTotal;
        private static Stopwatch _stopwatchLoop;

        //List of Best Values
        private static readonly List<double> BestValues = new List<double>();

        //Run MultipleOptimizationRuns(Entry point: Run RunOptimizationLoop (more than) once)
        public static void RunOptimizationLoopMultiple(object sender, DoWorkEventArgs e)
        {
            var logBaseName = LogName;
            var bestResult = new OptimizationResult(BolMaximize ? double.NegativeInfinity : double.PositiveInfinity, new List<decimal>(), 0, OptimizationResult.ResultType.Unknown);

            //Get worker and component
            _worker = sender as BackgroundWorker;
            _component = (OptimizationComponent)e.Argument;

            if (_component == null)
            {
                MessageBox.Show("FrOG Component not set to an object", "FrOG Error");
                return;
            }

            //Setup Variables
            _component.GhInOut_Instantiate();
            if (!_component.GhInOut.SetInputs() || !_component.GhInOut.SetOutput())
            {
                MessageBox.Show("Getting Variables and/or Objective failed", "Opossum Error");
                return;
            }

            //Main Loop
            var finishedRuns = 0;

            while (finishedRuns < Runs)
            {
                if (_worker == null || _worker.CancellationPending) break;

                //Log
                if (BolLog) LogName = logBaseName;
                if (BolRuns) LogName += $"_{finishedRuns + 1}";

                //Run RBFOpt
                var result = RunOptimizationLoop(_worker, PresetIndex);

                //Exit if there is no result
                if (result == null)
                    break;

                //Check is there is a better result
                if ((!BolMaximize && result.Value < bestResult.Value) || (BolMaximize && result.Value > bestResult.Value))
                    bestResult = result;

                //Very important to keep FrOG from crashing (probably needed to dispose the process in Run)
                System.Threading.Thread.Sleep(1000);

                finishedRuns++;
            }

            //Exit when there is no result
            if (double.IsPositiveInfinity(bestResult.Value) || double.IsNegativeInfinity(bestResult.Value)) return;

            //Set Grasshopper model to best value
            _component.GhInOut.NewSolution(bestResult.Parameters);

            //Show Result Message Box
            if (!BolRuns)
                MessageBox.Show(Log.GetResultString(bestResult, MaxIter, MaxIterNoProgress, MaxDuration), "FrOG Result");
            else
                MessageBox.Show($"Finished {finishedRuns} runs" + Environment.NewLine + $"Overall best value {bestResult.Value}", "FrOG Result");

            _worker?.CancelAsync();
        }

        //Run Solver (Main function)
        private static OptimizationResult RunOptimizationLoop(BackgroundWorker worker, int presetIndex)
        {
            _iterations = 0;
            _iterationsCurrentBest = 0;
            _bestValue = BolMaximize ? double.NegativeInfinity : double.PositiveInfinity;
            _bestParams = new List<decimal>();
            _resultType = OptimizationResult.ResultType.Unknown;

            //Get variables
            var variables = _component.GhInOut.Variables;
            //MessageBox.Show($"Parameter String: {variables}", "FrOG Parameters");

            //Stopwatches
            _stopwatchTotal = Stopwatch.StartNew();
            _stopwatchLoop = Stopwatch.StartNew();
            _stopwatchTotal.Start();

            //Clear Best Value List
            BestValues.Clear();

            //Prepare Solver
            if (worker.CancellationPending) return null;
            var solver = SolverList.GetSolverByIndex(PresetIndex);
            var preset = SolverList.GetPresetByIndex(presetIndex);

            //Prepare Log
            _log = BolLog ? new Log($"{Path.GetDirectoryName(_component.GhInOut.DocumentPath)}\\{LogName}.txt") : null;

            //Log Settings       
            _log?.LogSettings(preset);

            //Run Solver
            //MessageBox.Show("Starting Solver", "FrOG Debug");
            var bolSolverStarted = solver.RunSolver(variables, EvaluateFunction, preset, _component.GhInOut.ComponentFolder, _component.GhInOut.DocumentPath);

            if (!bolSolverStarted)
            {
                MessageBox.Show("Solver could not be started!");
                return null;
            }

            //Show Messagebox with RBFOpt error
            if (worker.CancellationPending)
                _resultType = OptimizationResult.ResultType.UserStopped;
            else if (_resultType == OptimizationResult.ResultType.SolverStopped || _resultType == OptimizationResult.ResultType.Unknown)
            {
                var strError = solver.GetErrorMessage();
                if (!string.IsNullOrEmpty(strError)) MessageBox.Show(strError, "FrOG Error");
            }

            //Result
            _stopwatchLoop.Stop();
            _stopwatchTotal.Stop();

            var result = new OptimizationResult(_bestValue, _bestParams, _iterations, _resultType);
            _log?.LogResult(result, _stopwatchTotal, MaxIter, MaxIterNoProgress, MaxDuration);

            return result;
        }

        public static double EvaluateFunction(IList<decimal> values)
        {
            _log?.LogIteration(_iterations + 1);
            //var strMessage = "Iteration " + _iterations + Environment.NewLine;
            //strMessage += $"Maximize: {BolMaximize}" + Environment.NewLine;
            //MessageBox.Show(strMessage);
            //MessageBox.Show("Variable Values: " + string.Join(" ",values));

            if (values == null)
            {
                _resultType = OptimizationResult.ResultType.SolverStopped;
                return double.NaN;
            }

            //Log Parameters
            _log?.LogParameters(string.Join(",", values), _stopwatchLoop);

            _stopwatchLoop.Reset();
            _stopwatchLoop.Start();

            //Run a new solution
            if (_worker.CancellationPending) return double.NaN;
            _component.GhInOut.NewSolution(values);

            //Evaluate Function
            var objectiveValue = _component.GhInOut.GetObjectiveValue();
            if (double.IsNaN(objectiveValue))
            {
                _resultType = OptimizationResult.ResultType.FrogStopped;
                return double.NaN;
            }

            _stopwatchLoop.Stop();

            //MessageBox.Show($"Function value: {objectiveValue}");

            //BolLog Solution
            _log?.LogFunctionValue(objectiveValue, _stopwatchLoop);

            _iterations += 1;
            _iterationsCurrentBest += 1;

            //Keep track of best value
            if ((!BolMaximize && objectiveValue < _bestValue) || (BolMaximize && objectiveValue > _bestValue))
            {
                _bestValue = objectiveValue;
                _bestParams = values;
                _iterationsCurrentBest = 0;
            }

            BestValues.Add(_bestValue);

            //Report Best Values
            _worker.ReportProgress(0, BestValues);

            //BolLog Minimum
            _log?.LogCurrentBest(_bestParams, _bestValue, _stopwatchTotal, _iterationsCurrentBest);

            //Optimization Results
            //No Improvement
            if (BolMaxIterNoProgress && _iterationsCurrentBest >= MaxIterNoProgress)
            {
                _resultType = OptimizationResult.ResultType.NoImprovement;
                return double.NaN;
            }
            //Maximum Evaluations reached

            MessageBox.Show($"{BolMaxIter} {_iterations}/{MaxIter}");

            if (BolMaxIter && _iterations >= MaxIter)
            {
                _resultType = OptimizationResult.ResultType.MaximumEvals;
                return double.NaN;
            }
            //Maximum Duration reached
            if (BolMaxDuration && _stopwatchTotal.Elapsed.TotalSeconds >= MaxDuration)
            {
                _resultType = OptimizationResult.ResultType.MaximumTime;
                return double.NaN;
            }
            //Else: Pass result to Solver
            _stopwatchLoop.Reset();
            _stopwatchLoop.Start();

            if (BolMaximize) objectiveValue = -objectiveValue;
            return objectiveValue;
        }
    }
}
