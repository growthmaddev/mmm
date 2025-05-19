import { useRef, useEffect } from "react";
import * as d3 from "d3";

interface CurvePoint {
  spend: number;
  response: number;
}

interface DataPoint {
  channel: string;
  current: number;
  recommended: number;
  curve: CurvePoint[];
  color: string;
}

interface ResponseCurveChartProps {
  data: DataPoint[];
}

export default function ResponseCurveChart({ data }: ResponseCurveChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data.length) return;
    
    // Clear previous chart
    d3.select(svgRef.current).selectAll("*").remove();
    
    // Generate some curve data if not provided
    const processedData = data.map(channel => {
      if (!channel.curve || channel.curve.length === 0) {
        // Generate a curve with diminishing returns
        const curve: CurvePoint[] = [];
        const maxSpend = channel.current * 1.5;
        
        for (let spend = 0; spend <= maxSpend; spend += maxSpend / 20) {
          // Simple diminishing returns function: response = a * ln(spend + 1)
          // Where 'a' is calibrated to make the curve pass through the current point
          const a = channel.current * 0.8 / Math.log(channel.current + 1);
          const response = a * Math.log(spend + 1);
          curve.push({ spend, response });
        }
        
        return { ...channel, curve };
      }
      
      return channel;
    });
    
    // Chart dimensions
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create SVG group
    const svg = d3.select(svgRef.current)
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => d.current * 1.5) || 0])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => 
        d3.max(d.curve, c => c.response) || 0
      ) || 0])
      .range([innerHeight, 0]);
    
    // Add X axis
    svg.append("g")
      .attr("transform", `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .ticks(5)
        .tickFormat(d => `$${d3.format(".0s")(d as number)}`))
      .selectAll("text")
        .style("font-size", "10px");
    
    // Add Y axis
    svg.append("g")
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll("text")
        .style("font-size", "10px");
    
    // X axis label
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + margin.bottom - 5)
      .style("font-size", "10px")
      .style("fill", "#64748b")
      .text("Marketing Spend");
    
    // Y axis label
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("transform", `translate(${-margin.left + 10}, ${innerHeight / 2}) rotate(-90)`)
      .style("font-size", "10px")
      .style("fill", "#64748b")
      .text("Response (Sales)");
    
    // Add the response curves
    const line = d3.line<CurvePoint>()
      .x(d => xScale(d.spend))
      .y(d => yScale(d.response))
      .curve(d3.curveMonotoneX);
    
    // Draw curves
    processedData.forEach((channel, i) => {
      // Add the path
      svg.append("path")
        .datum(channel.curve)
        .attr("fill", "none")
        .attr("stroke", channel.color)
        .attr("stroke-width", 2)
        .attr("d", line);
      
      // Add current spend marker
      svg.append("circle")
        .attr("cx", xScale(channel.current))
        .attr("cy", yScale(channel.curve.find(c => c.spend >= channel.current)?.response || 0))
        .attr("r", 4)
        .attr("fill", channel.color)
        .attr("stroke", "white")
        .attr("stroke-width", 1);
      
      // Add recommended spend marker if different from current
      if (Math.abs(channel.recommended - channel.current) > 0.01 * channel.current) {
        svg.append("circle")
          .attr("cx", xScale(channel.recommended))
          .attr("cy", yScale(channel.curve.find(c => c.spend >= channel.recommended)?.response || 0))
          .attr("r", 4)
          .attr("fill", "none")
          .attr("stroke", channel.color)
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", "2,2");
      }
    });
    
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
      .attr("transform", `translate(${innerWidth - 100}, 0)`);
    
    processedData.forEach((d, i) => {
      const legendRow = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);
      
      legendRow.append("line")
        .attr("x1", 0)
        .attr("y1", 10)
        .attr("x2", 20)
        .attr("y2", 10)
        .attr("stroke", d.color)
        .attr("stroke-width", 2);
      
      legendRow.append("text")
        .attr("x", 25)
        .attr("y", 15)
        .attr("font-size", "10px")
        .style("text-transform", "capitalize")
        .text(d.channel);
    });
    
    // Add interaction layer
    svg.append("rect")
      .attr("width", innerWidth)
      .attr("height", innerHeight)
      .attr("opacity", 0)
      .on("mousemove", function(event) {
        const [mouseX] = d3.pointer(event);
        const spendValue = xScale.invert(mouseX);
        
        // Find closest data points
        const tooltipData = processedData.map(channel => {
          const closestPoint = channel.curve.reduce((prev, curr) => 
            Math.abs(curr.spend - spendValue) < Math.abs(prev.spend - spendValue) ? curr : prev
          );
          
          return {
            channel: channel.channel,
            color: channel.color,
            spend: closestPoint.spend,
            response: closestPoint.response
          };
        });
        
        // Update tooltip content
        tooltip
          .style("opacity", 1)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px")
          .html(`
            <strong>Spend: $${d3.format(",")(Math.round(spendValue))}</strong>
            <div style="margin-top: 4px;">
              ${tooltipData.map(d => `
                <div style="display: flex; align-items: center; margin-top: 2px;">
                  <div style="width: 8px; height: 8px; background-color: ${d.color}; margin-right: 5px;"></div>
                  <div style="text-transform: capitalize;">${d.channel}: ${d3.format(",.0f")(d.response)}</div>
                </div>
              `).join('')}
            </div>
          `);
        
        // Show vertical line at mouse position
        svg.selectAll(".mouse-line").remove();
        svg.append("line")
          .attr("class", "mouse-line")
          .attr("x1", mouseX)
          .attr("x2", mouseX)
          .attr("y1", 0)
          .attr("y2", innerHeight)
          .attr("stroke", "#475569")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "3,3");
      })
      .on("mouseout", function() {
        // Hide tooltip and guide line
        tooltip.style("opacity", 0);
        svg.selectAll(".mouse-line").remove();
      });
    
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
