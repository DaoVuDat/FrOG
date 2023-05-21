using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FrOG.Utils
{
    public class RandomDistributions : Random
    {
        public RandomDistributions()
            : base()
        { }

        /// <summary>
        /// Normal distributed random number.
        /// </summary>
        /// <param name="mean">Mean of the distribution.</param>
        /// <param name="stdDev">Standard deviation of the distribution.</param>
        /// <returns>Normal distributed random number.</returns>
        public double NextGaussian(double mean, double stdDev)
        {
            //Random rand = new Random(); //reuse this if you are generating many
            double u1 = base.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = base.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)

            return randNormal;

        }
    }
}
