classdef Gsat < handle
    % Gsat: satellite position/velocity/clock data
    % ---------------------------------------------------------------------
    % Gsat Declaration:
    % obj = Gsat()
    % ---------------------------------------------------------------------
    % Gsat Properties:
    %   n         : 1x1, Number of epochs
    %   nsat      : 1x1, Number of satellites
    %   sat       : 1x(obj.nsat), Satellite number defined in RTKLIB
    %   prn       : 1x(obj.nsat), Satellite prn/slot number
    %   sys       : 1x(obj.nsat), Satellite system (SYS_GPS, SYS_GLO, ...)
    %   satstr    : 1x(obj.nsat), Satellite id cell array ('Gnn','Rnn','Enn','Jnn','Cnn','Inn' or 'nnn')
    %   time      : 1x1, Time, gt.Gtime class
    %   x         : (obj.n)x(obj.nsat), satellite position in ECEF X (m)
    %   y         : (obj.n)x(obj.nsat), satellite position in ECEF Y (m)
    %   z         : (obj.n)x(obj.nsat), satellite position in ECEF Z (m)
    %   vx        : (obj.n)x(obj.nsat), satellite position in ECEF X (m/s)
    %   vy        : (obj.n)x(obj.nsat), satellite position in ECEF Y (m/s)
    %   vz        : (obj.n)x(obj.nsat), satellite position in ECEF Z (m/s)
    %   dts       : (obj.n)x(obj.nsat), satellite clock bias (m)
    %   ddts      : (obj.n)x(obj.nsat), satellite clock drift (m/s)
    %   var       : (obj.n)x(obj.nsat), satellite position and clock error variance (m^2)
    %   svh       : (obj.n)x(obj.nsat), satellite health flag
    %   pos       : 1x1, Receiver position, gt.Gpos class
    %   rng       : (obj.n)x(obj.nsat), geometric distance (m)
    %   ex        : (obj.n)x(obj.nsat), line-of-sight vector in ECEF X (m)
    %   ey        : (obj.n)x(obj.nsat), line-of-sight vector in ECEF Y (m)
    %   ez        : (obj.n)x(obj.nsat), line-of-sight vector in ECEF Z (m)
    %   az        : (obj.n)x(obj.nsat), satellite azimuth angle (deg)
    %   el        : (obj.n)x(obj.nsat), satellite elevation angle (deg)
    %   trp       : (obj.n)x(obj.nsat), tropospheric path delay
    %   ionL1     : (obj.n)x(obj.nsat), Ionospheric delay at L1 frequency
    %   ionL2     : (obj.n)x(obj.nsat), Ionospheric delay at L2 frequency
    %   ionL5     : (obj.n)x(obj.nsat), Ionospheric delay at L5 frequency
    % ---------------------------------------------------------------------
    % Gsat Methods:
    %   setSatObs(gobs, gnav, ephopt): set satellite data at observation time
    %   setSat(gtime, sat, gnav, ephopt): set satellite data at input time
    %   setRcvPos(gpos): set satellite data based on receiver position
    %   setRcvVel(gvel): set satellite data based on receiver velocity
    %   setRcvPosVel(obj, gpos, gvel): set satellite data based on receiver position/velocity
    %   gsat = copy(obj): Copy object
    %   gsat = select(obj, tidx, sidx): select from time/satellite index
    %   gsat = selectSat(obj, sidx): select from satellite index
    %   gsat = selectTime(obj, tidx): select from time index
    %   gsat = selectTimeSpan(obj, ts, te): select from time
    %   [refidx,Dinv] = referenceSat(obj, tidx): compute reference satellite
    %   plotSky(obj, tidx, sidx): plot satellite azimuth and elevation angles
    %   update_azel(obj, sld, txt, ps, uniquesys): Update azimuth and elevation
    %   dtr = roundDateTime(~, dt, dec) : round datetime  
    %   help()
    % ---------------------------------------------------------------------
    %     Author: Taro Suzuki

    properties
        n % Number of epochs
        nsat % Number of satellites
        sat % Satellite number defined in RTKLIB
        prn % Satellite prn/slot number
        sys % Satellite system (SYS_GPS, SYS_GLO, ...)
        satstr % Satellite id cell array ('Gnn','Rnn','Enn','Jnn','Cnn','Inn' or 'nnn')
        time % Time, gt.Gtime class
        x % satellite position in ECEF X (m)
        y % satellite position in ECEF Y (m)
        z % satellite position in ECEF Z (m)
        vx % satellite position in ECEF X (m/s)
        vy % satellite position in ECEF Y (m/s)
        vz % satellite position in ECEF Z (m/s)
        dts % satellite clock bias (m)
        ddts % satellite clock drift (m/s)
        var % satellite position and clock error variance (m^2)
        svh % satellite health flag
        pos % Receiver position, gt.Gpos class
        vel % satellite velocity
        rng % geometric distance (m)
        rate %  range rate
        ex % line-of-sight vector in ECEF X (m)
        ey % line-of-sight vector in ECEF Y (m)
        ez % line-of-sight vector in ECEF Z (m)
        az % satellite azimuth angle (deg)
        el; % satellite elevation angle (deg)
        trp; % tropospheric path delay
        ionL1; % Ionospheric delay at L1 frequency
        ionL2; % Ionospheric delay at L2 frequency
        ionL5; % Ionospheric delay at L5 frequency
        ionL6; % Ionospheric delay at L6 frequency
        ionL7; % Ionospheric delay at L7 frequency
        ionL8; % Ionospheric delay at L8 frequency
        ionL9; % Ionospheric delay at L9 frequency
    end
    properties(Access=private)
        obs,nav;
        FTYPE = ["L1","L2","L5","L6","L7","L8","L9"];
    end
    methods
        %% constractor
        function obj = Gsat(varargin) 
            if nargin==0
                % generate empty class instance
                obj.n = 0;
                obj.nsat = 0;
            elseif nargin==2
                obj.setSatObs(varargin{1},varargin{2}); % satposs
            elseif nargin==3 && isa(varargin{1}, 'gt.Gobs')
                obj.setSatObs(varargin{1},varargin{2},varargin{3}); % satposs
            elseif nargin==3 && size(varargin{1},2)==6
                obj.setSat(varargin{1},varargin{2},varargin{3}); % satpos
            elseif nargin==4
                obj.setSat(varargin{1},varargin{2},varargin{3},varargin{4}); % satpos
            else
                error('Wrong input arguments');
            end
        end

        %% compute satellite data at observation time
        function setSatObs(obj, gobs, gnav, ephopt)
            % setSatObs: set satellite data at observation time
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            % obj.setSatObs(gobs, gnav, ephopt)
            %
            % Input: ------------------------------------------------------
            % gobs : GNSS RINEX ovservation data class
            % gnav : GNSS RINEX navigation data class
            % ephopt : 1×1 : broadcast ephemeris
            % 
            arguments
                obj gt.Gsat
                gobs gt.Gobs
                gnav gt.Gnav
                ephopt (1,1) = gt.C.EPHOPT_BRDC
            end
            if isenum(ephopt)
                ephopt = double(ephopt);
            end
            [obj.x,obj.y,obj.z,obj.vx,obj.vy,obj.vz,obj.dts,obj.ddts,obj.var,obj.svh] ...
                = rtklib.satposs(gobs.struct, gnav.struct, ephopt);
            obj.n = gobs.n;
            obj.nsat = gobs.nsat;
            obj.sat = gobs.sat;
            obj.prn = gobs.prn;
            obj.sys = gobs.sys;
            obj.satstr = gobs.satstr;
            obj.time = gobs.time;
            obj.obs = gobs;
            obj.nav = gnav;
        end

        %% compute satellite data at input time
        function setSat(obj, gtime, sat, gnav, ephopt)
            % setSat: set satellite data at input time
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.setSat(gtime, sat, gnav, ephopt)
            %
            % Input: ------------------------------------------------------
            %   gtime : 1x1, gt.Gtime class
            %   sat : 1x(obj.nsat), Satellite number defined in RTKLIB
            %   gnav : GNSS RINEX navigation data class
            %   ephopt : Ephemeris Option Broadcast Ephemeris
            % 
            arguments
                obj gt.Gsat
                gtime gt.Gtime
                sat (1,:) {mustBeInteger, mustBeVector}
                gnav gt.Gnav
                ephopt (1,1) = gt.C.EPHOPT_BRDC
            end
            if isenum(ephopt)
                ephopt = double(ephopt);
            end
            [obj.x,obj.y,obj.z,obj.vx,obj.vy,obj.vz,obj.dts,obj.ddts,obj.var,obj.svh] ...
                = rtklib.satpos(gtime.ep, sat, gnav.struct, ephopt);
            obj.n = gtime.n;
            obj.nsat = length(sat);
            obj.sat = sat;
            [sys_, obj.prn] = rtklib.satsys(obj.sat);
            obj.sys = gt.C.SYS(sys_);
            obj.satstr = rtklib.satno2id(obj.sat);
            obj.time = gtime;
            obj.obs = gobs;
            obj.nav = gnav;
        end

        %% compute satellite data based on receiver position
        function setRcvPos(obj, gpos)
            % setRcvPos: set satellite data based on receiver position
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            % obj.setRcvPos(gpos)
            % 
            % Input: ------------------------------------------------------
            % gpos : Receiver position
            % 
            arguments
                obj gt.Gsat
                gpos gt.Gpos
            end
            if gpos.n ~= obj.n && gpos.n ~= 1
                error('The size of gpos is equal to the size of obj.n or 1');
            end
            obj.pos = gpos;

            % ex,ey,ez,rng,az,el
            [obj.rng, obj.ex, obj.ey, obj.ez] ...
                = rtklib.geodist(obj.x, obj.y, obj.z, gpos.xyz);
            [obj.az, obj.el] = rtklib.satazel(gpos.llh, obj.ex, obj.ey, obj.ez);

            % trop,iono
            obj.trp = rtklib.tropmodel(obj.obs.time.ep,gpos.llh,obj.az,obj.el);
            if ~isfield(obj.obs.L1,"freq")
                obj.obs.setFrequencyFromNav(obj.nav);
            end
            for f = obj.FTYPE
                if ~isempty(obj.obs.(f))
                    obj.("ion"+f) = rtklib.ionmodel(obj.obs.time.ep,obj.nav.ion.gps,gpos.llh,obj.az,obj.el,obj.obs.(f).freq);
                end
            end
        end

        %% compute satellite data based on receiver velocity
        function setRcvVel(obj, gvel)
            % setRcvVel: set satellite data based on receiver velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            % obj.setRcvPos(gvel)
            % 
            % Input: ------------------------------------------------------
            % gvel : receiver velocity
            % 
            arguments
                obj gt.Gsat
                gvel gt.Gvel
            end
            if gvel.n ~= obj.n && gvel.n ~= 1
                error('The size of gvel is equal to the size of obj.n or 1');
            end
            if isempty(obj.pos)
                error('setRcvPos must be called first');
            end
            obj.vel = gvel;
            
            % range rate
            vsrx = obj.vx-obj.vel.xyz(:,1);
            vsry = obj.vy-obj.vel.xyz(:,2);
            vsrz = obj.vz-obj.vel.xyz(:,3);
            
            % relative velocity + sagnac effect
            obj.rate = vsrx.*obj.ex+vsry.*obj.ey+vsrz.*obj.ez+...
                gt.C.OMGE/gt.C.CLIGHT*(obj.vy.*obj.pos.xyz(:,1)+obj.y.*gvel.xyz(:,1)-obj.vx.*obj.pos.xyz(:,2)-obj.x.*gvel.xyz(:,2));
        end

        %% compute satellite data based on receiver position/velocity
        function setRcvPosVel(obj, gpos, gvel)
            % setRcvPosVel: set satellite data based on receiver position/velocity
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.setRcvPosVel(gpos, gvel)
            % 
            % Input: ------------------------------------------------------
            %   gpos: Receiver position
            %   gvel: Receiver velocity
            % 
            arguments
                obj gt.Gsat
                gpos gt.Gpos
                gvel gt.Gvel
            end
            obj.setRcvPos(gpos);
            obj.setRcvVel(gvel);
        end

        %% copy
        function gsat = copy(obj)
            % copy: Copy object
            % -------------------------------------------------------------
            % MATLAB handle class is used, so if you want to create a
            % different instance, you need to use the copy method.
            %
            % Usage: ------------------------------------------------------
            %   gtime = obj.copy()
            %
            % Output: ------------------------------------------------------
            %   gsat : satellite position/velocity/clock data
            %
            arguments
                obj gt.Gsat
            end
            gsat = obj.select(1:obj.n,1:obj.nsat);
        end

        %% select from time/satellite index
        function gsat = select(obj, tidx, sidx)
            % select: select from time/satellite index
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.select(tidx, sidx)
            % 
            % Input: ------------------------------------------------------
            %   indx : time index
            %   sidx : satellite index
            % 
            % Output: ------------------------------------------------------
            %   gsat : satellite position/velocity/clock data
            % 
            arguments
                obj gt.Gsat
                tidx {mustBeInteger, mustBeVector}
                sidx {mustBeInteger, mustBeVector}
            end
            if ~any(tidx)
                error('Selected time index is empty');
            end
            if ~any(sidx)
                error('Selected satellite index is empty');
            end
            gsat = gt.Gsat();
            gsat.time = obj.time.select(tidx);
            gsat.n = gsat.time.n;
            gsat.sat = obj.sat(sidx);
            gsat.prn = obj.prn(sidx);
            gsat.sys = obj.sys(sidx);
            gsat.satstr = obj.satstr(sidx);
            gsat.nsat = length(obj.sat);

            gsat.x = obj.x(tidx, sidx);
            gsat.y = obj.y(tidx, sidx);
            gsat.z = obj.z(tidx, sidx);
            gsat.vx = obj.vx(tidx, sidx);
            gsat.vy = obj.vy(tidx, sidx);
            gsat.vz = obj.vz(tidx, sidx);
            gsat.dts = obj.dts(tidx, sidx);
            gsat.ddts = obj.ddts(tidx, sidx);
            gsat.var = obj.var(tidx, sidx);
            gsat.svh = obj.svh(tidx, sidx);
            % receiver position
            if ~isempty(obj.pos)
                gsat.pos = obj.pos;
                gsat.rng = obj.rng(tidx, sidx);
                gsat.ex = obj.ex(tidx, sidx);
                gsat.ey = obj.ey(tidx, sidx);
                gsat.ez = obj.ez(tidx, sidx);
                gsat.az = obj.az(tidx, sidx);
                gsat.el = obj.el(tidx, sidx);                
            end
            % receiver velocity
            if ~isempty(obj.vel)
                gsat.vel = obj.vel;
                gsat.rate = obj.rate(tidx, sidx);
            end
            if ~isempty(obj.obs)
                gsat.obs = obj.obs.select(tidx, sidx);
            end
            if ~isempty(obj.nav)
                gsat.nav = obj.nav;
            end
        end

        %% select from satellite index
        function gsat = selectSat(obj, sidx)
            % selectSat: select from satellite index
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.selectSat(sidx)
            % 
            % Input: ------------------------------------------------------
            %   sidx : satellite index
            % 
            % Output: ------------------------------------------------------
            %   gsat : satellite position/velocity/clock data
            arguments
                obj gt.Gsat
                sidx {mustBeInteger, mustBeVector}
            end
            gsat = obj.select(1:obj.n, sidx);
        end

        %% select from time index
        function gsat = selectTime(obj, tidx)
            % selectTime: select from time index
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %    obj.selectTime(tidx)
            % 
            % Input: ------------------------------------------------------
            %   tidx : time index
            % 
            % Output: ------------------------------------------------------
            %   gsat : satellite position/velocity/clock data
            arguments
                obj gt.Gsat
                tidx {mustBeInteger, mustBeVector}
            end
            gsat = obj.select(tidx, 1:obj.nsat);
        end

        %% select from time
        function gsat = selectTimeSpan(obj, ts, te)
            % selectTimeSpan: select from time
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %    obj.selectTimeSpan(ts, te)
            % 
            % Input: ------------------------------------------------------
            %   ts : 1x1, gt.Gtime, Start time
            %   te : 1x1, gt.Gtime, End time
            % 
            % Output: ------------------------------------------------------
            %   gsat : satellite position/velocity/clock data
            arguments
                obj gt.Gsat
                ts gt.Gtime
                te gt.Gtime
            end
            ndec = floor(-log10(obj.time.estInterval()));
            tr = obj.roundDateTime(obj.time.t,ndec);
            tsr = obj.roundDateTime(ts.t,ndec);
            ter = obj.roundDateTime(te.t,ndec);
            tidx = tr>=tsr & tr<=ter;
            gsat = obj.selectTime(tidx);
        end

        %% compute reference satellite
        function [refidx,Dinv] = referenceSat(obj, tidx)
            % referenceSat: compute reference satellite
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.referenceSat(tidx)
            % Input: ------------------------------------------------------
            %   tidx : Logical or numeric index to select
            % 
            % Output: ------------------------------------------------------
            %   refidx : Reference satellite index
            %   Dinv : Matrix based on each reference satellite
            arguments
                obj gt.Gsat
                tidx {mustBeInteger, mustBeVector} = 1:obj.n
            end
            if isempty(obj.pos)
                error('Call obj.setRcvPos(gpos) first to set the receiver position');
            end
            nsys_prev = 0;
            Dinv = [];
            g = findgroups(obj.sys);
            for gu=unique(g)
                nsys = nnz(g==gu);
                elref = mean(obj.el(tidx,:),1,"omitmissing");
                elref(g~=gu) = 0;
                if any(elref)
                    [~, satidx] = max(elref);
                    Dgu = eye(nsys);
                    Dgu(:,satidx-nsys_prev) = -1;
                    Dgu(satidx-nsys_prev,:) = -1;
                    Dinv = blkdiag(Dinv,inv(Dgu));
                    refsatidx(gu) = satidx;
                else
                    refsatidx(gu) = nsys_prev+1;
                    Dinv = blkdiag(Dinv,eye(nsys));
                end
                nsys_prev = nsys_prev+nsys;
            end
            refidx = refsatidx(g);
        end

        %% skyplot (using navigation toolbox)
        % function plotSky(obj)
        %     if isempty(obj.pos)
        %         error('Call obj.setRcvPos(gpos) first to set the receiver position');
        %     end
        % 
        %     uniquesys = unique(obj.sys);
        %     col = gt.C.C_SYS(uniquesys,:);
        %     satsys = categorical(obj.sys);
        %     fig = figure;
        %     sp = skyplot(obj.az(idx,:),obj.el(idx,:),obj.satstr,'GroupData',satsys);
        %     set(gca,'ColorOrder',col)
        %     set(gca,'LabelFontSize',10)
        %     legend(string(gt.C.SYSNAME(uniquesys)),'Location','WestOutside','FontSize',10);
        % 
        %     txt = uicontrol(fig,'Style','text','Position',[40 40 100 40]);
        %     txt.String = string(obj.time.t(obj.n),'yyyy-MM-dd HH:mm:ss.S');
        % 
        %     sld = uicontrol(fig,'Style','slider','Position',[40 80 100 20]);
        %     sld.Callback = @(src,event)obj.update_azel(sld,txt,sp);
        %     sld.Value = obj.n;
        %     sld.Min = 1;
        %     sld.Max = obj.n;
        %     sld.SliderStep = [1 10]/obj.n;
        % end

        %% skyplot
        function plotSky(obj, tidx, sidx)
            % plotSky: plot satellite azimuth and elevation angles
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.plotSky(tidx, sidx)
            %
            % Input: ------------------------------------------------------
            %   tidx : Logical or numeric index to select
            %   sidx : Logical or numeric index to select
            arguments
                obj gt.Gsat
                tidx {mustBeInteger, mustBeVector} = 1:obj.n
                sidx {mustBeInteger, mustBeVector} = 1:obj.nsat
            end
            if isempty(obj.pos)
                error('Call obj.setRcvPos(gpos) first to set the receiver position');
            end
            gsat = obj.select(tidx, sidx);

            uniquesys = unique(gsat.sys);
            col = gt.C.C_SYS(uniquesys,:);

            tippp = [dataTipTextRow('Satellite','');
                   dataTipTextRow('Azimuth','ThetaData');
                   dataTipTextRow('Elevation','RData')];
            tipps = [dataTipTextRow('Satellite','ColorVariable');
                   dataTipTextRow('Azimuth','ThetaData');
                   dataTipTextRow('Elevation','RData')];

            fig = figure;
            for i=1:length(uniquesys)
                isat = gsat.sys==uniquesys(i);
                az_ = gsat.az(:,isat)/180*pi;
                el_ = gsat.el(:,isat);
                satstr_ = gsat.satstr(isat);
                if gsat.n > 1
                    pp = polarplot(az_,el_,'Color',col(i,:),'linewidth',1);
                else
                    pp = polarplot(az_,el_,'.','Color',col(i,:),'MarkerSize',0.1);
                end
                hold on;
                for j=1:length(pp)
                    pp(j).DataTipTemplate.DataTipRows = tippp;
                    pp(j).DataTipTemplate.DataTipRows(1).Label = satstr_{j};
                end
                ps(i) = polarscatter(az_(1,:),el_(1,:),'filled','SizeData',150,'MarkerEdgeColor',col(i,:),'MarkerFaceColor',col(i,:),'MarkerFaceAlpha',0.5);
                ps(i).ColorVariable = satstr_;
                ps(i).DataTipTemplate.DataTipRows = tipps;
                tx{i} = text(az_(1,:),el_(1,:)-10,satstr_,'Color',col(i,:),'FontSize',10,'HorizontalAlignment','center');
            end
            ax = gca;
            ax.RLim = [0 90];
            ax.RDir = 'reverse';
            ax.ThetaDir = 'clockwise';
            ax.ThetaZeroLocation = 'top';
            ax.RAxisLocation = 270;
            ax.ThetaTickLabel(1) = {'N'};
            ax.ThetaTickLabel(4) = {'E'};
            ax.ThetaTickLabel(7) = {'S'};
            ax.ThetaTickLabel(10) = {'W'};
            ax.RTickLabel = {'0°','20°','40°','60°','80°'};
            legend(ps, string(gt.C.SYSNAME(double(uniquesys))),'Location','WestOutside','FontSize',10);

            % text and slidar control
            if gsat.n > 1
                txt = uicontrol(fig,'Style','text','Position',[40 40 100 40]);
                txt.String = string(gsat.time.t(1),'yyyy-MM-dd HH:mm:ss.S');
                sld = uicontrol(fig,'Style','slider','Position',[40 80 100 20]);
                sld.Callback = @(src,event)gsat.update_azel(sld,txt,ps,uniquesys);
                sld.Value = 1;
                sld.Min = 1;
                sld.Max = gsat.n;
                sld.SliderStep = [1 10]/gsat.n;
            end
        end

        %% help
        function help(~)
            doc gt.Gsat
        end

    end

    methods(Access=private)
        % function update_azel(obj, sld, txt, sp)
        %     idx = uint32(sld.Value);
        %     set(sp,AzimuthData=obj.az(1:idx,:), ElevationData=obj.el(1:idx,:));
        %     txt.String = string(obj.time.t(idx),'yyyy-MM-dd HH:mm:ss.S');
        % end
        function update_azel(obj, sld, txt, ps, uniquesys)
            % update_azel: Update azimuth and elevation
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.update_azel(tidx, sld, txt, ps, uniquesys)
            %
            % Input: ------------------------------------------------------
            %   sld : An object for specifying the time index
            %   txt : A text object for displaying the time
            %   ps : Azimuth and elevation alignment
            %   uniquesys :  List of satellites observed
            idx = uint32(sld.Value);
            txt.String = string(obj.time.t(idx),'yyyy-MM-dd HH:mm:ss.S');
            for i=1:length(ps)
                isat = obj.sys==uniquesys(i);
                az_ = obj.az(:,isat)/180*pi;
                el_ = obj.el(:,isat);
                ps(i).ThetaData = az_(idx,:);
                ps(i).RData = el_(idx,:);
            end
        end

        % round datetime
        function dtr = roundDateTime(~, dt, dec)
            % roundDateTime: round datetime
            % -------------------------------------------------------------
            %
            % Usage: ------------------------------------------------------
            %   obj.roundDateTime(dt, dec)
            % 
            % Input: ------------------------------------------------------
            %   dt : Target time data
            %   dec : Roud precision
            % 
            % Output: ------------------------------------------------------
            %   dtr: Rounded time
            %
            dtr = dateshift(dt,'start','minute') + seconds(round(second(dt),dec));
        end
    end
end