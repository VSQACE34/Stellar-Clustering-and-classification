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
          <h1 className="Welcome-text text-[12px]">picture of stellar objects</h1>
        </div>
      </div>

      <div className="relative z-[20] mt-[10rem] left-[5rem]">
        <div className="text-[20px] text-left font-medium text-gray-300 ">
          The whys and hows of the classification model are shown here:
        </div>
      </div>
    </div>
  );
};

export default Insight;
