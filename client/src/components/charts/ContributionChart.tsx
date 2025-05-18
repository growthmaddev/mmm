import { useRef, useEffect } from "react";
import * as d3 from "d3";

interface DataPoint {
  channel: string;
  contribution: number;
  color: string;
}

interface ContributionChartProps {
  data: DataPoint[];
}

export default function ContributionChart({ data }: ContributionChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    
    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove();
    
    // Chart dimensions
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 10, right: 10, bottom: 30, left: 10 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const radius = Math.min(innerWidth, innerHeight) / 2;
    
    // Create SVG group
    const svg = d3.select(svgRef.current)
      .append("g")
      .attr("transform", `translate(${width/2}, ${height/2})`);
    
    // Prepare data for pie chart
    const pie = d3.pie<DataPoint>()
      .value(d => d.contribution * 100)
      .sort(null);
    
    const arcs = pie(data);
    
    // Create pie slices
    const arcGenerator = d3.arc<d3.PieArcDatum<DataPoint>>()
      .innerRadius(radius * 0.5)  // Creates a donut chart
      .outerRadius(radius * 0.9);
    
    // Add slices
    svg.selectAll("path")
      .data(arcs)
      .join("path")
      .attr("d", arcGenerator)
      .attr("fill", d => d.data.color)
      .attr("stroke", "white")
      .style("stroke-width", 2)
      .style("opacity", 0.9)
      .on("mouseover", function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .style("opacity", 1)
          .attr("transform", `scale(1.05)`);
        
        // Show tooltip
        tooltip
          .transition()
          .duration(200)
          .style("opacity", 0.9);
        
        tooltip
          .html(`
            <strong>${d.data.channel.charAt(0).toUpperCase() + d.data.channel.slice(1)}</strong><br>
            ${(d.data.contribution * 100).toFixed(1)}% of sales
          `)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this)
          .transition()
          .duration(200)
          .style("opacity", 0.9)
          .attr("transform", `scale(1)`);
        
        // Hide tooltip
        tooltip
          .transition()
          .duration(500)
          .style("opacity", 0);
      });
    
    // Add center text
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", "14px")
      .attr("fill", "#1e293b")
      .text("Channel");
      
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.5em")
      .attr("font-size", "12px")
      .attr("fill", "#64748b")
      .text("Contribution");
    
    // Add labels
    const labelArc = d3.arc<d3.PieArcDatum<DataPoint>>()
      .innerRadius(radius * 0.9 + 10)
      .outerRadius(radius * 0.9 + 10);
    
    // Create tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "chart-tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("padding", "8px")
      .style("background", "white")
      .style("border-radius", "4px")
      .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
      .style("pointer-events", "none")
      .style("font-size", "12px")
      .style("z-index", "10");
    
    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${-radius}, ${radius + 10})`);
    
    // Only show legend if there's space
    if (height > 200) {
      data.forEach((d, i) => {
        const legendRow = legend.append("g")
          .attr("transform", `translate(0, ${i * 20})`);
        
        legendRow.append("rect")
          .attr("width", 10)
          .attr("height", 10)
          .attr("fill", d.color);
        
        legendRow.append("text")
          .attr("x", 15)
          .attr("y", 9)
          .attr("font-size", "10px")
          .style("text-transform", "capitalize")
          .text(d.channel);
      });
    }
    
    // Cleanup
    return () => {
      tooltip.remove();
    };
  }, [data]);
  
  return (
    <svg 
      ref={svgRef} 
      width="100%" 
      height="100%" 
      viewBox={`0 0 ${svgRef.current?.clientWidth || 400} ${svgRef.current?.clientHeight || 250}`}
      preserveAspectRatio="xMidYMid meet"
    />
  );
}
