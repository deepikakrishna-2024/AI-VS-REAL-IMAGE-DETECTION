import { motion } from "framer-motion";
import { Brain, AlertTriangle, BarChart3, Cpu } from "lucide-react";

const steps = [
  {
    icon: Cpu,
    title: "Image Preprocessing",
    desc: "The uploaded image is resized to 224×224 pixels and normalized for consistent analysis.",
  },
  {
    icon: Brain,
    title: "Neural Network Analysis",
    desc: "A deep learning model (MobileNetV2-based) examines pixel patterns, textures, and artifacts typical of AI generation.",
  },
  {
    icon: BarChart3,
    title: "Confidence Scoring",
    desc: "The model outputs a probability score indicating how likely the image is real or AI-generated.",
  },
  {
    icon: AlertTriangle,
    title: "Important Caveats",
    desc: "AI detection is probabilistic — confidence scores reflect likelihood, not absolute certainty. Results should be treated as one signal among many.",
  },
];

const HowItWorks = () => (
  <section className="w-full max-w-2xl mx-auto mt-20">
    <motion.h2
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      className="text-2xl font-bold font-display text-center text-gradient-primary mb-10"
    >
      How It Works
    </motion.h2>

    <div className="grid gap-6">
      {steps.map((step, i) => (
        <motion.div
          key={step.title}
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.1 }}
          className="flex gap-4 rounded-lg border border-border bg-card p-5"
        >
          <div className="shrink-0 rounded-md bg-secondary p-2.5">
            <step.icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-display font-semibold text-foreground mb-1">
              {step.title}
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {step.desc}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  </section>
);

export default HowItWorks;
