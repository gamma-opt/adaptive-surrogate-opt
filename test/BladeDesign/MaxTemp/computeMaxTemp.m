function [max_temp] = computeMaxTemp(Tair, Tgas, hair, ...
    hgaspressureside, hgassuctionside, hgastip)
    % Create a static structural model
    tmodel = createpde("thermal","steadystate");
    % Import and plot the geometry, displaying face labels
    importGeometry(tmodel,"Blade.stl");
    % Generate a mesh with the maximum element size 0.01
    tmodel.Mesh = generateMesh(tmodel,"Hmax",0.01);
    % Determine the temperature distribution and compute the maximum 
    % temperature of the blade
    kapp = 11.5; % in W/m/K
    thermalProperties(tmodel,"ThermalConductivity",kapp);
    % Interior cooling
    thermalBC(tmodel,"Face",[15 12 14], ...
                     "ConvectionCoefficient",hair, ...  % hair
                     "AmbientTemperature",Tair);  % T_air = 150
    % Pressure side
    thermalBC(tmodel,"Face",11, ...
                     "ConvectionCoefficient",hgaspressureside, ... % h_gas_pressureside = 50
                     "AmbientTemperature",Tgas); % T_gas = 1000
    % Suction side             
    thermalBC(tmodel,"Face",10, ...
                     "ConvectionCoefficient",hgassuctionside, ... % h_gas_suctionside = 40
                     "AmbientTemperature",Tgas); % T_gas = 1000
    % Tip
    thermalBC(tmodel,"Face",13, ...
                     "ConvectionCoefficient",hgastip, ... % h_gas_tip = 20
                     "AmbientTemperature",Tgas); % T_gas = 1000
    % Base (exposed to hot gases)
    thermalBC(tmodel,"Face",1, ...
                     "ConvectionCoefficient",40, ...
                     "AmbientTemperature",800);
    % Root in contact with hot gases
    thermalBC(tmodel,"Face",[6 9 8 2 7], ...
                     "ConvectionCoefficient",15, ...
                     "AmbientTemperature",400);
    % Root in contact with metal
    thermalBC(tmodel,"Face",[3 4 5], ...
                     "ConvectionCoefficient",1000, ...
                     "AmbientTemperature",300);
    % Solve the thermal model
    Rt = solve(tmodel);
    max_temp = max(Rt.Temperature);
end