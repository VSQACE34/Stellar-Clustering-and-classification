"use client";
import React from "react";

import { motion } from "framer-motion";
import { slideInFromTop } from "@/utils/motion";
import Image from "next/image";

const Insight = () => {
  return (
    <div className="flex flex-row relative justify-center min-h-screen w-full h-full" id="insight">
      <div className="absolute w-auto h-auto top-0 z-[5]">
        <motion.div
          variants={slideInFromTop}
          className="text-[40px] font-medium text-center text-gray-200"
        >
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-cyan-500">
            {" "}
            Model{" "}
          </span>
          Insights 
        </motion.div>
      </div>
      
      <div className="flex flex-col items-center justify-center translate-y-[-50px] relative z-[20] w-auto h-auto">
        <div className="flex flex-col items-right group cursor-pointer w-auto h-auto left-[10px]">
          <Image
            src="/stellar.webp"
            alt="Lock Main"
            width={650}
            height={600}
            className=" z-10"
          />
        </div>
        <div className="Welcome-box px-[15px] py-[4px] z-[20] brder my-[20px] border-[#7042f88b] opacity-[0.9]">
          <h1 className="Welcome-text text-[12px]">picture of stellar objects (from left:- star, galaxy, qso)</h1>
        </div>
      </div>

      <div className="relative z-[20] mt-[10rem] left-[5rem]">
        <div className="text-[14px] text-left font-medium text-gray-400 my-[-2] max-w-[600px]">
        As depicted in the image, stars, quasars (QSOs), and galaxies exhibit inherent differences in size and luminous intensity, the latter being the quantity of light an object emits. Stars, although plentiful, are less luminous and become almost imperceptible at great distances. In contrast, galaxies, due to their substantial size, remain visible over larger distances. Quasars, despite being smaller than galaxies, are exceptionally luminous, making them detectable even further away. These differences are quantified by an object's redshift—the greater the redshift, the further away and more luminous the object is, with quasars exhibiting the highest redshifts. To accurately classify these celestial bodies, the model utilizes not just redshift data, but also alpha (right ascension) and delta (declination) coordinates, which provide a precise celestial location. Using XGBoost, the model enhances its classification accuracy by focusing on objects that are more challenging to classify. K-Means clustering further refines this process by organizing the objects into clusters based on the similarity of these features. Thus, the model not only classifies objects based on their light-emitting properties but also spatially groups them according to their positions in the celestial sphere, revealing the significant role these coordinates play in understanding the cosmos.
        </div>
      </div>
    </div>
  );
};

export default Insight;
