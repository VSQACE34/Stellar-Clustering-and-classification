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
          As can be seen from the image on the left, there exists a fundamental difference in stars, 
          quasars (qso) and galaxies. The difference being the size and the luminous intensity which
          is the amount of light emitted by the object. Fundamentally speaking, stars are smaller 
          than Galaxies. But Quasars, however, are smaller than galaxies but are the brightest object 
          in the universe. So, as a result the light from the stars, get dimmer as they get farther and
          at one point it becomes almost invisible. Galaxies being bigger than stars are visible for a 
          far greater range but quasars are so bright that they can be seen even further away from the 
          galaxies. This qunatity of seeing an object is shown in the redshift of the object. Stars 
          being smaller and less luminous than galaxies or quasars have smaller redshift. While galaxies
          and quasars have higher redshift with Quasar having the highest redshift. The feature importance
          of the features are shown below
        </div>
      </div>
    </div>
  );
};

export default Insight;
