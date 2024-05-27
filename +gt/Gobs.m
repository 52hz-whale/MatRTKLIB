classdef Gobs < handle
    % Gobs: GNSS RINEX ovservation data class
    %
    % Gobs Declaration:
    % obj = Gobs(file)
    %   file      : 1x1, RINEX observation file
    %
    % obj = Gobs(obsstr)
    %   obsstr    : 1x1, RTKLIB observation struct
    %
    % Gobs Properties:
    %   n         : 1x1, Number of epochs
    %   nsat      : 1x1, Number of satellites
    %   sat       : 1x(obj.nsat), Satellite number defined in RTKLIB
    %   prn       : 1x(obj.nsat), Satellite prn/slot number
    %   sys       : 1x(obj.nsat), Satellite system (SYS_GPS, SYS_GLO, ...)
    %   satstr    : 1x(obj.nsat), Satellite id cell array ('Gnn','Rnn','Enn','Jnn','Cnn','Inn' or 'nnn')
    %   time      : 1x1, Time, gt.Gtime class
    %   dt        : 1x1, Observation time interval (s)
    %   pos       : 1x1, Position in RINEX header, gt.Gpos class
    %   glofcn    : 1x(obj.nsat), Frequency channel number for GLONASS 
    %   L1        : 1x1, L1 observation struct
    %     .P      : (obj.n)x(obj.nsat), Pseudorange (m)
    %     .L      : (obj.n)x(obj.nsat), Carrier phase (cycle)
    %     .D      : (obj.n)x(obj.nsat), Doppler (Hz)
    %     .S      : (obj.n)x(obj.nsat), SNR (dB-Hz)
    %     .I      : (obj.n)x(obj.nsat), LLI flag
    %     .ctype  : 1x(obj.nsat), Observation code cell array {'1C', '1P', '1P',...}
    %     .freq   : 1x(obj.nsat), Carrier frequency (Hz)
    %     .lam    : 1x(obj.nsat), Wavelength (m)
    %   L2        : 1x1, L2 observation struct
    %   L5        : 1x1, L5 observation struct
    %   L6        : 1x1, L6 observation struct
    %   L7        : 1x1, L7 observation struct
    %   L8        : 1x1, L8 observation struct
    %   L9        : 1x1, L9 observation struct
    % 
    % Gobs Methods:
    %   setObsFile(file):
    %   setObsStruct(obsstr):
    %   setFrequency():
    %   setFrequencyFromNav(nav):
    %   outObs(file):
    %   append(gobs):
    %   difference(gobj)
    %   gobs = select(tidx, sidx):
    %   gobs = selectSat(sidx):
    %   gobs = selectTime(tidx):
    %   gobs = selectTimeSpan(ts, te, [dt])
    %   obsstr = struct([tidx], [sidx])
    %   gobs = fixedInterval([dt]):
    %   [gobs, gobsref] = common(gobsref):
    %   [gobs, gobsref] = commonSat(gobsref):
    %   [gobs, gobsref] = commonTime(gobsref):
    %   gobsSD = singleDifference(gobs):
    %   plot([freq], [sidx]):
    %   plotNSat([freq], [snrth], [sidx]):
    %   plotSky(nav, [sidx]):
    %   help()
    %
    %     Author: Taro Suzuki

    properties
        n, nsat, sat, prn, sys, satstr, time, dt, pos, glofcn, L1, L2, L5, L6, L7, L8, L9, Lwl, Lml, Lif;
    end
    properties(Access=private)
        FTYPE = ["L1","L2","L5","L6","L7","L8","L9","Lwl","Lml","Lif"];
    end
    methods
        %% constractor
        function obj = Gobs(varargin)
            if nargin==0
                % generate empty class instance
                obj.n = 0;
                obj.nsat = 0;                
            elseif nargin==1 && (ischar(varargin{1}) || isStringScalar(varargin{1}))
                obj.setObsFile(char(varargin{1})); % file
            elseif nargin==1 && isstruct(varargin{1})
                obj.setObsStruct(varargin{1}); % obs struct
            else
                error('Wrong input arguments');
            end
        end

        %% set observation from RINEX file
        function setObsFile(obj, file)
            arguments
                obj gt.Gobs
                file (1,:) char
            end
            try
                [obs, basepos, fcn] = rtklib.readrnxobs(file);
            catch
                error('Wrong RINEX observation file: %s',file);
            end

            % pos
            if ~all(basepos==[0,0,0])
                obj.pos = gt.Gpos(basepos,'xyz');
            end
            % glofcn
            idxglo = obs.sys==gt.C.SYS_GLO;
            fcn(fcn==0) = NaN;
            obj.glofcn = NaN(1,obs.nsat);
            obj.glofcn(idxglo) = fcn(obs.prn(idxglo))-8;
            
            obj.setObsStruct(obs);
        end
        %% set observation from observation struct
        function setObsStruct(obj, obsstr)
            arguments
                obj gt.Gobs
                obsstr (1,1) struct
            end
            ep = obsstr.ep;
            obj.n = size(obsstr.ep,1);
            obj.nsat = size(obsstr.sat,2);
            obj.sat = obsstr.sat;
            [sys_, obj.prn] = rtklib.satsys(obj.sat);
            obj.sys = gt.C.SYS(sys_);
            obj.satstr = rtklib.satno2id(obj.sat);
            obj.time = gt.Gtime(ep);
            obj.dt = obj.time.estInterval();
            for f = obj.FTYPE
                if isfield(obsstr,f)
                    obj.(f) = obsstr.(f);
                end
            end
            if isempty(obj.glofcn)
                obj.glofcn = NaN(1,obj.nsat);
            end
            if ~all(isnan(obj.glofcn))
                obj.setFrequency();
            end
        end
        %% set carrier frequency
        function setFrequency(obj)
            arguments
                obj gt.Gobs
            end
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    code = rtklib.obs2code(obj.(f).ctype);
                    obj.(f).freq = rtklib.code2freq(double(obj.sys), code, obj.glofcn);
                    obj.(f).freq(obj.(f).freq==0) = NaN;
                    obj.(f).lam = gt.C.CLIGHT./obj.(f).freq;
                end
            end
        end

        %% set carrier frequency from navigation data
        function setFrequencyFromNav(obj, nav)
            arguments
                obj gt.Gobs
                nav (1,1)
            end
            if ~isstruct(nav)
                if isa(nav, 'gt.Gnav')
                    nav = nav.struct();
                else
                    error('Input must be nav struct of gt.Gnav');
                end
            end
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(obj.(f),"ctype")
                        code = rtklib.obs2code(obj.(f).ctype);
                        obj.(f).freq = rtklib.sat2freq(obj.sat,code,nav);
                        obj.(f).freq(obj.(f).freq==0) = NaN;
                        obj.(f).lam = gt.C.CLIGHT./obj.(f).freq;
                    end
                end
            end
        end

        %% output observation file
        function outObs(obj, file)
            arguments
                obj gt.Gobs
                file (1,:) char
            end
            obsstr = obj.struct();

            fcn = zeros(1,32);
            xyz = zeros(1,3);

            % GLONASS FCN for RINEX header            
            if ~any(isnan(obj.glofcn))
                sysglo = obj.sys==gt.C.SYS_GLO;
                fcn(obj.prn(sysglo)) = obj.glofcn(sysglo)+8;
            end
            if ~isempty(obj.pos)
                xyz = obj.pos.xyz;
            end
            rtklib.outrnxobs(file, obsstr, xyz, fcn);
        end

        %% append
        function append(obj, gobs)
            arguments
                obj gt.Gobs
                gobs gt.Gobs
            end
            obsstr.n = obj.n+gobs.n;
            obsstr.sat = unique([obj.sat, gobs.sat]);
            obsstr.nsat = length(obsstr.sat);
            [obsstr.sys, obsstr.prn] = rtklib.satsys(obsstr.sat);
            obsstr.satstr = rtklib.satno2id(obsstr.sat);
            obsstr.ep = [obj.time.ep; gobs.time.ep];
            obsstr.tow = [obj.time.tow; gobs.time.tow];
            obsstr.week = [obj.time.week; gobs.time.week];
            [~,sidx1] = intersect(obsstr.sat, obj.sat);
            [~,sidx2] = intersect(obsstr.sat, gobs.sat);
            for f = obj.FTYPE
                if ~isempty(obj.(f)) || ~isempty(gobs.(f))
                    obsstr.(f) = obj.initFreqStruct(f,obsstr.n,obsstr.nsat);
                    if ~isempty(obj.(f))
                        obsstr.(f) = obj.setFreqStruct(obsstr.(f),obj.(f),1:obj.n,1:obj.n,sidx1,1:obj.nsat);
                    end
                    if ~isempty(gobs.(f))
                        obsstr.(f) = obj.setFreqStruct(obsstr.(f),gobs.(f),(obj.n+1):obsstr.n,1:gobs.n,sidx2,1:gobs.nsat);
                    end
                end
            end
            obj.setObsStruct(obsstr);
        end
        
        %% pseudorange mask
        function gobs = maskP(obj, mask, freq)
            arguments
                obj gt.Gobs
                mask logical
                freq string = obj.FTYPE 
            end
            if size(mask,1)~=obj.n || size(mask,2)~=obj.nsat
                error('mask array size does not match');
            end
            gobs = obj.copy();
            for f = freq
                if ~isempty(gobs.(f))
                    gobs.(f).P(mask) = NaN;
                    if isfield(gobs.(f),'resP')
                        gobs.(f).resP(mask) = NaN;
                        gobs.(f).resPc(mask) = NaN;
                    end
                    if isfield(gobs.(f),'Pd')
                        gobs.(f).Pd(mask) = NaN;
                        if isfield(gobs.(f),'resPd')
                            gobs.(f).resPd(mask) = NaN;
                        end
                    end
                    if isfield(gobs.(f),'Pdd')
                        gobs.(f).Pdd(mask) = NaN;
                        if isfield(gobs.(f),'resPdd')
                            gobs.(f).resPdd(mask) = NaN;
                        end
                    end
                end
            end
        end

        %% pseudorange mask
        function gobs = maskD(obj, mask, freq)
            arguments
                obj gt.Gobs
                mask logical
                freq string = obj.FTYPE 
            end
            if size(mask,1)~=obj.n || size(mask,2)~=obj.nsat
                error('mask array size does not match');
            end
            gobs = obj.copy();
            for f = freq
                if ~isempty(gobs.(f))
                    gobs.(f).D(mask) = NaN;
                    if isfield(gobs.(f),'resD')
                        gobs.(f).resD(mask) = NaN;
                        gobs.(f).resDc(mask) = NaN;
                    end
                    if isfield(gobs.(f),'Dd')
                        gobs.(f).Dd(mask) = NaN;
                        if isfield(gobs.(f),'resDd')
                            gobs.(f).resDd(mask) = NaN;
                        end
                    end
                end
            end
        end

        %% carrier phase mask
        function gobs = maskL(obj, mask, freq)
            arguments
                obj gt.Gobs
                mask logical
                freq string = obj.FTYPE 
            end
            if size(mask,1)~=obj.n || size(mask,2)~=obj.nsat
                error('mask array size does not match');
            end
            gobs = obj.copy();
            for f = freq
                if ~isempty(gobs.(f))
                    gobs.(f).L(mask) = NaN;
                    if isfield(gobs.(f),'resL')
                        gobs.(f).resL(mask) = NaN;
                        gobs.(f).resLc(mask) = NaN;
                    end
                    if isfield(gobs.(f),'Ld')
                        gobs.(f).Ld(mask) = NaN;
                        if isfield(gobs.(f),'resLd')
                            gobs.(f).resLd(mask) = NaN;
                        end
                    end
                    if isfield(gobs.(f),'Ldd')
                        gobs.(f).Ldd(mask) = NaN;
                        if isfield(gobs.(f),'resLdd')
                            gobs.(f).resLdd(mask) = NaN;
                        end
                    end
                end
            end
        end

        %% applay mask to observation 
        function gobs = mask(obj, mask, freq)
            arguments
                obj gt.Gobs
                mask logical
                freq string = obj.FTYPE 
            end
            if size(mask,1)~=obj.n || size(mask,2)~=obj.nsat
                error('mask array size does not match the observations');
            end
            gobs = obj.copy();
            gobs = gobs.maskP(mask,freq);
            gobs = gobs.maskL(mask,freq);
            gobs = gobs.maskD(mask,freq);
        end

        %% applay mask from LLI flag 
        function gobs = maskLLI(obj)
            arguments
                obj gt.Gobs
            end
            gobs = obj.copy();
            for f = gobs.FTYPE
                if ~isempty(gobs.(f))
                    mask = gobs.(f).I>=1;
                    gobs = gobs.maskL(mask,f);
                end
            end
        end

        %% eliminate all NaN satellite
        function gobs = eliminateNaN(obj)
            arguments
                obj gt.Gobs
            end
            sidx = false(size(obj.sat));
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    sidx = sidx | any(~isnan(obj.(f).P));
                end
            end
            gobs = obj.selectSat(sidx);
        end

        %% copy
        function gobs = copy(obj)
            arguments
                obj gt.Gobs
            end
            gobs = obj.select(1:obj.n,1:obj.nsat);
        end

        %% select from time/satellite index
        function gobs = select(obj, tidx, sidx)
            arguments
                obj gt.Gobs
                tidx {mustBeInteger, mustBeVector}
                sidx {mustBeInteger, mustBeVector}
            end
            if ~any(tidx)
                warning('Selected time index is empty');
                gobs = gt.Gobs();
                return
            end
            if ~any(sidx)
                warning('Selected satellite index is empty');
                gobs = gt.Gobs();
                return
            end
            obsstr = obj.struct(tidx, sidx);
            gobs = gt.Gobs(obsstr);

            gobs.pos = obj.pos;
            gobs.glofcn = obj.glofcn(sidx);
            obj.copyFrequency(gobs,1:gobs.nsat,sidx);
            obj.copyAdditinalObservation(gobs,1:gobs.n,tidx,1:gobs.nsat,sidx);
        end

        %% select from satellite index
        function gobs = selectSat(obj, sidx)
            arguments
                obj gt.Gobs
                sidx {mustBeInteger, mustBeVector}
            end
            gobs = obj.select(1:obj.n, sidx);
        end

        %% select from time index
        function gobs = selectTime(obj, tidx)
            arguments
                obj gt.Gobs
                tidx {mustBeInteger, mustBeVector}
            end
            gobs = obj.select(tidx, 1:obj.nsat);
        end

        %% select from time
        function gobs = selectTimeSpan(obj, ts, te)
            arguments
                obj gt.Gobs
                ts gt.Gtime
                te gt.Gtime
            end
            tr = obj.roundDateTime(obj.time.t,2);
            tsr = obj.roundDateTime(ts.t,2);
            ter = obj.roundDateTime(te.t,2);
            tidx = tr>=tsr & tr<=ter;
            gobs = obj.selectTime(tidx);
        end

        %% convert to struct
        function obsstr = struct(obj, tidx, sidx)
            arguments
                obj gt.Gobs
                tidx {mustBeInteger, mustBeVector} = 1:obj.n
                sidx {mustBeInteger, mustBeVector} = 1:obj.nsat
            end
            obsstr.sat = obj.sat(sidx);
            obsstr.prn = obj.prn(sidx);
            obsstr.sys = double(obj.sys(sidx));
            obsstr.satstr = obj.satstr(sidx);
            obsstr.ep = obj.time.ep(tidx,:);
            obsstr.tow = obj.time.tow(tidx);
            obsstr.week = obj.time.week(tidx);
            obsstr.n = size(obsstr.ep,1);
            obsstr.nsat = size(obsstr.sat,2);
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    obsstr.(f) = obj.selectFreqStruct(obj.(f), tidx, sidx);
                end
            end
        end

        %% fixed interval
        function gobs = fixedInterval(obj, dt)
            arguments
                obj gt.Gobs
                dt (1,1) double = 0
            end
            if dt==0; dt = obj.dt; end
            tr = obj.roundDateTime(obj.time.t,2);
            tfixr = obj.roundDateTime((tr(1):seconds(dt):tr(end))',2);
            nfix = length(tfixr);
            tfix = NaT(nfix,1);
            [~, idx1,idx2] = intersect(tfixr,tr);
            tfix(idx1) = obj.time.t(idx2);
            tfix = fillmissing(tfix,'linear');
            gtfix = gt.Gtime(tfix);

            obsstr.n = nfix;
            obsstr.nsat = obj.nsat;
            obsstr.sat = obj.sat;
            obsstr.prn = obj.prn;
            obsstr.sys = double(obj.sys);
            obsstr.satstr = obj.satstr;
            obsstr.ep = gtfix.ep;
            obsstr.tow = gtfix.tow;
            obsstr.week = gtfix.week;

            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    obsstr.(f) = obj.initFreqStruct(f,obsstr.n,obsstr.nsat);
                    obsstr.(f) = obj.setFreqStruct(obsstr.(f),obj.(f),idx1,idx2,1:obj.nsat,1:obj.nsat);
                end
            end
            gobs = gt.Gobs(obsstr);
            gobs.pos = obj.pos;
            gobs.glofcn = obj.glofcn;
        end

        %% common obsevation
        function [gobsc, gobrefc] = commonObs(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            [gobsc, gobrefc] = obj.commonSat(gobsref);
            [gobsc, gobrefc] = gobsc.commonTime(gobrefc);
        end

        %% common satellite
        function [gobsc, gobrefc] = commonSat(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            [~,sidx1,sidx2] = intersect(obj.sat,gobsref.sat);
            gobsc = obj.selectSat(sidx1);
            gobrefc = gobsref.selectSat(sidx2);
        end

        %% common time
        function [gobsc, gobrefc] = commonTime(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            ndec = floor(-log10(obj.dt));
            t = obj.roundDateTime(obj.time.t, ndec);
            ndecref = floor(-log10(gobsref.dt));
            tref = obj.roundDateTime(gobsref.time.t, ndecref);
            [~,tidx1,tidx2] = intersect(t,tref);
            gobsc = obj.selectTime(tidx1);
            gobrefc = gobsref.selectTime(tidx2);
        end

        %% same obsevation
        function gobs = sameObs(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            gobs = obj.sameSat(gobsref);
            gobs = gobs.sameTime(gobsref);
        end

        %% same satellite
        function gobs = sameSat(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            [~,sidx1,sidx2] = intersect(gobsref.sat,obj.sat);
            obsstr.n = obj.n;
            obsstr.nsat = gobsref.nsat;
            obsstr.sat = gobsref.sat;
            obsstr.prn = gobsref.prn;
            obsstr.sys = double(gobsref.sys);
            obsstr.satstr = gobsref.satstr;
            obsstr.ep = obj.time.ep;
            obsstr.tow = obj.time.tow;
            obsstr.week = obj.time.week;

            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    obsstr.(f) = obj.initFreqStruct(f,obsstr.n,obsstr.nsat);
                    obsstr.(f) = obj.setFreqStruct(obsstr.(f),obj.(f),1:obj.n,1:obj.n,sidx1,sidx2);
                end
            end
            gobs = gt.Gobs(obsstr);
            gobs.pos = obj.pos;
            gobs.glofcn = gobsref.glofcn;
            obj.copyAdditinalObservation(gobs,1:gobs.n,1:gobs.n,sidx1,sidx2);
        end

        %% same time
        function gobs = sameTime(obj, gobsref)
            arguments
                obj gt.Gobs
                gobsref gt.Gobs
            end
            ndec = floor(-log10(obj.dt));
            t = obj.roundDateTime(obj.time.t, ndec);
            ndecref = floor(-log10(gobsref.dt));
            tref = obj.roundDateTime(gobsref.time.t, ndecref);

            [~,tidx1,tidx2] = intersect(tref,t);
            obsstr.n = gobsref.n;
            obsstr.nsat = obj.nsat;
            obsstr.sat = obj.sat;
            obsstr.prn = obj.prn;
            obsstr.sys = double(obj.sys);
            obsstr.satstr = obj.satstr;
            obsstr.ep = gobsref.time.ep;
            obsstr.tow = gobsref.time.tow;
            obsstr.week = gobsref.time.week;

            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    obsstr.(f) = obj.initFreqStruct(f,obsstr.n,obsstr.nsat);
                    obsstr.(f) = obj.setFreqStruct(obsstr.(f),obj.(f),tidx1,tidx2,1:obj.nsat,1:obj.nsat);
                end
            end
            gobs = gt.Gobs(obsstr);
            gobs.pos = obj.pos;
            gobs.glofcn = gobsref.glofcn;
            obj.copyAdditinalObservation(gobs,tidx1,tidx2,1:gobs.nsat,1:gobs.nsat);
        end

        %% carrier smoothing
        function gobs = carrierSmooth(obj)
            arguments
                obj gt.Gobs
            end
            gobs = obj.copy();
            % for f = obj.FTYPE
            %     if isfield(obj.(f),'resP')
            %         dresPL = obj.(f).resP-obj.(f).resL;
            %         idx_slip = obj.(f).I>0;
            %         if obj.dt<=1.0
            %            idx_slip = idx_slip |...
            %                [false; abs(obj.(f).dDL)>2.0] | ...
            %                [false; abs(gobs.(f).dDP)>20.0];
            %         end
            %         idx_slip(1:10:end,:) = true;
            %         group_slip = cumsum(idx_slip);
            %         resPs = NaN(obj.n,obj.nsat);
            %         for j=1:obj.nsat
            %             meandresPL = splitapply(@nanmean,dresPL(:,j),group_slip(:,j));
            %             resPs(:,j) = obj.(f).resL(:,j)+meandresPL(group_slip(:,j));
            %         end
            %         obj.(f).resPs = resPs;
            %     end
            % end
        end
        %% linear combination
        function gobs = linearCombination(obj)
            arguments
                obj gt.Gobs
            end
            gobs = obj.copy();
            
            % middle-lane (L1-L5)
            if ~isempty(obj.L1) && ~isempty(obj.L5)
                gobs.Lml.freq = obj.L1.freq-obj.L5.freq;
                gobs.Lml.lam = gt.C.CLIGHT./gobs.Lml.freq;
                gobs.Lml.L = obj.L1.L-obj.L5.L;
            end
        end
        %% residuals
        function gobs = residuals(obj, gsat)
            arguments
                obj gt.Gobs
                gsat gt.Gsat
            end
            if obj.nsat ~= gsat.nsat
                error('obj.nsat and gsat.nsat must be the same')
            end
            if obj.n ~= gsat.n
                error('obj.n and gsat.n must be the same')
            end
            if isempty(gsat.pos)
                error('Call gsat.setRcvPos(gpos) first to set the receiver position');
            end
            if isempty(gsat.vel)
                error('Call gsat.setRcvVel(gvel) first to set the receiver velocity');
            end
            gobs = obj.copy();
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(gobs.(f),"P"); gobs.(f).resP = gobs.(f).P-gsat.rng+gsat.dts; end % pseudorange residuals
                    if isfield(gobs.(f),"L"); gobs.(f).resL = gobs.(f).L.*gobs.(f).lam-gsat.rng+gsat.dts; end % carrier phase residuals
                    if isfield(gobs.(f),"D"); gobs.(f).resD = -gobs.(f).D.*gobs.(f).lam-gsat.rate+gsat.ddts; end % doppler residuals
                    
                    if isprop(gsat,"ion"+f)
                        if ~isempty(gsat.("ion"+f))
                            if isfield(gobs.(f),"P"); gobs.(f).resPc = gobs.(f).P-(gsat.rng-gsat.dts+gsat.("ion"+f)+gsat.trp); end % pseudorange residuals
                            if isfield(gobs.(f),"L"); gobs.(f).resLc = gobs.(f).L.*gobs.(f).lam-(gsat.rng-gsat.dts-gsat.("ion"+f)+gsat.trp); end % carrier phase residuals
                            if isfield(gobs.(f),"D"); gobs.(f).resDc = -gobs.(f).D.*gobs.(f).lam-(gsat.rate-gsat.ddts); end % doppler residuals
                        end
                    end
                end
            end
        end
        
        %% single diffenrece
        function gobsSD = singleDifference(obj, gobs)
            arguments
                obj gt.Gobs
                gobs gt.Gobs
            end
            if obj.nsat ~= gobs.nsat
                error('obj.nsat and gobs.nsat must be the same')
            end
            if obj.n ~= gobs.n
                error('obj.n and gobs.n must be the same')
            end
            gobsSD = obj.copy();
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(obj.(f),"P") && isfield(gobs.(f),"P"); gobsSD.(f).Pd = obj.(f).P-gobs.(f).P; end
                    if isfield(obj.(f),"L") && isfield(gobs.(f),"L"); gobsSD.(f).Ld = obj.(f).L-gobs.(f).L; end
                    if isfield(obj.(f),"D") && isfield(gobs.(f),"D"); gobsSD.(f).Dd = obj.(f).D-gobs.(f).D; end
                    if isfield(obj.(f),"resP") && isfield(gobs.(f),"resP"); gobsSD.(f).resPd = obj.(f).resP-gobs.(f).resP; end
                    if isfield(obj.(f),"resL") && isfield(gobs.(f),"resL"); gobsSD.(f).resLd = obj.(f).resL-gobs.(f).resL; end
                    if isfield(obj.(f),"resD") && isfield(gobs.(f),"resD"); gobsSD.(f).resDd = obj.(f).resD-gobs.(f).resD; end
                end
            end
        end

        %% double diffenrece
        function gobsDD = doubleDifference(obj, refidx)
            arguments
                obj gt.Gobs
                refidx {mustBeInteger, mustBeVector}
            end
            if length(refidx) ~= obj.nsat
                error('Size of refidx must be obj.nsat')
            end
            
            gobsDD = obj.copy();
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(obj.(f),"Pd"); gobsDD.(f).Pdd = obj.(f).Pd-obj.(f).Pd(:,refidx); end
                    if isfield(obj.(f),"Ld"); gobsDD.(f).Ldd = obj.(f).Ld-obj.(f).Ld(:,refidx); end
                    if isfield(obj.(f),"resPd"); gobsDD.(f).resPdd = obj.(f).resPd-obj.(f).resPd(:,refidx); end
                    if isfield(obj.(f),"resLd"); gobsDD.(f).resLdd = obj.(f).resLd-obj.(f).resLd(:,refidx); end
                end
            end
        end

        %% plot
        function plot(obj, freq, sidx)
            arguments
                obj gt.Gobs
                freq (1,2) char {mustBeMember(freq,{'L1','L2','L5','L6','L7','L8','L9'})} = 'L1'
                sidx {mustBeInteger, mustBeVector} = 1:obj.nsat
            end
            gobs = obj.selectSat(sidx);
            if isempty(gobs.(freq))
                warning([freq ': No observations'])
            else
                f = figure;
                f.Position(2) = f.Position(2)-f.Position(4);
                f.Position(4) = 2*f.Position(4);
                y = gobs.nsat:-1:1;
                for i=1:gobs.nsat
                    scatter(gobs.time.t,y(i)*ones(gobs.n,1),[],gobs.(freq).S(:,i),'filled');
                    hold on;
                end
                grid on;
                xlim([gobs.time.t(1) gobs.time.t(end)]);
                ylim([0 gobs.nsat+1]);

                yticks(1:gobs.nsat)
                yticklabels(fliplr(gobs.satstr));
                c = colorbar(gca,'northoutside');
                c.Label.String = [freq ' SNR (dB-Hz)'];
                drawnow
            end
        end
        function plotNSat(obj, freq, snrth, sidx)
            arguments
                obj gt.Gobs
                freq (1,2) char {mustBeMember(freq,{'L1','L2','L5','L6','L7','L8','L9'})} = 'L1'
                snrth (1,1) double = 0.0
                sidx {mustBeInteger, mustBeVector} = 1:obj.nsat
            end
            gobs = obj.selectSat(sidx);
            if isempty(gobs.(freq))
                warning([freq ': No observations'])
            else
                figure;
                satsyss = unique(gobs.sys);
                col = gt.C.C_SYS(satsyss,:);
                for i=1:length(satsyss)
                    isys = gobs.sys==satsyss(i);
                    nsys(:,i) = sum(gobs.(freq).S(:,isys)>snrth,2);
                end
                area(gobs.time.t,nsys);
                grid on;
                set(gca,'ColorOrder',col);
                legend(string(gt.C.SYSNAME(double(satsyss))));
                xlim([gobs.time.t(1) gobs.time.t(end)]);
                title(['Number of ' freq ' observations (CNR threhould ' num2str(snrth) ' dB-Hz)']);
                drawnow
            end
        end
        function plotSky(obj, gnav, tidx, sidx)
            arguments
                obj gt.Gobs
                gnav gt.Gnav
                tidx {mustBeInteger, mustBeVector} = 1:obj.n
                sidx {mustBeInteger, mustBeVector} = 1:obj.nsat
            end
            if isempty(obj.pos)
                error('Apploximated position in RINEX Header is empty');
            end
            gobs = obj.select(tidx, sidx);
            gsat = gt.Gsat(gobs, gnav);
            gsat.setRcvPos(obj.pos);
            gsat.plotSky;
        end
        
        %% help
        function help(~)
            doc gt.Gobs
        end

        %% overload
        % minus: single difference
        function gobs = minus(obj, gobs)
            arguments
                obj gt.Gobs
                gobs gt.Gobs
            end
            gobs = obj.singleDifference(gobs);
        end
    end

    methods(Access=private)
        % round datetime
        function dtr = roundDateTime(~, dt, dec)
            dtr = dateshift(dt,'start','minute') + seconds(round(second(dt),dec));
        end

        % select LLI
        function Isel = selectLLI(~,I,tind,sind)
            I(isnan(I)) = 0;

            I1 = I==1 | I==3;
            I2 = I==2 | I==3;

            I1sum = cumsum(I1);
            I1sum_sel = I1sum(tind,sind);
            [n_sel,nsat_sel] = size(I1sum_sel);
            if n_sel>1
                I1sel = [zeros(1,nsat_sel);diff(I1sum_sel,1)];
                I1sel(I1sel>=1) = 1;
            else
                I1sel = I1sum_sel;
                I1sel(I1sel>=1) = 1;
            end
            I2sel = 2*I2(tind,sind);

            Isel = I1sel+I2sel;
        end

        % select observation
        function Fsel = selectFreqStruct(obj,F,tidx,sidx)
            if isfield(F,"P"); Fsel.P = F.P(tidx,sidx); end
            if isfield(F,"L"); Fsel.L = F.L(tidx,sidx); end
            if isfield(F,"D"); Fsel.D = F.D(tidx,sidx); end
            if isfield(F,"S"); Fsel.S = F.S(tidx,sidx); end
            if isfield(F,"I"); Fsel.I = obj.selectLLI(F.I,tidx,sidx); end
            if isfield(F,"ctype"); Fsel.ctype = F.ctype(sidx); end
            if isfield(F,'freq'); Fsel.freq = F.freq(sidx);end
            if isfield(F,'lam'); Fsel.lam = F.lam(sidx); end
        end

        % initialize observation
        function Fini = initFreqStruct(obj,f,n,nsat)
            if isfield(obj.(f),"P"); Fini.P = NaN(n,nsat); end
            if isfield(obj.(f),"L"); Fini.L = NaN(n,nsat); end
            if isfield(obj.(f),"D"); Fini.D = NaN(n,nsat); end
            if isfield(obj.(f),"S"); Fini.S = NaN(n,nsat); end
            if isfield(obj.(f),"I"); Fini.I = NaN(n,nsat); end
            if isfield(obj.(f),"ctype"); Fini.ctype = repmat({''},1,nsat); end
            if isfield(obj.(f),'freq'); Fini.freq = NaN(1,nsat); end
            if isfield(obj.(f),'lam'); Fini.lam = NaN(1,nsat); end
        end

        % set observation
        function Fset = setFreqStruct(obj,Fset,F,tidx1,tidx2,sidx1,sidx2)
            if isfield(F,"P"); Fset.P(tidx1,sidx1) = F.P(tidx2,sidx2); end
            if isfield(F,"L"); Fset.L(tidx1,sidx1) = F.L(tidx2,sidx2); end
            if isfield(F,"D"); Fset.D(tidx1,sidx1) = F.D(tidx2,sidx2); end
            if isfield(F,"S"); Fset.S(tidx1,sidx1) = F.S(tidx2,sidx2); end
            if isfield(F,"I"); Fset.I(tidx1,sidx1) = obj.selectLLI(F.I,tidx2,sidx2); end
            if isfield(F,"ctype"); Fset.ctype(sidx1) = F.ctype(sidx2); end
            if isfield(F,'freq'); Fset.freq(sidx1) = F.freq(sidx2); end
            if isfield(F,'lam'); Fset.lam(sidx1) = F.lam(sidx2); end
        end

        % copy frequency and wavelength
        function copyFrequency(obj,dst,sidx1,sidx2)
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(f,'freq')
                        dst.(f).freq(sidx1) = obj.(f).freq(sidx2);
                        dst.(f).lam(sidx1) = obj.(f).lam(sidx2);
                    end
                end
            end
        end

        % copy frequency and wavelength
        function copyAdditinalObservation(obj,dst,tidx1,tidx2,sidx1,sidx2)
            for f = obj.FTYPE
                if ~isempty(obj.(f))
                    if isfield(obj.(f),'resP'); dst.(f).resP(tidx1,sidx1) = obj.(f).resP(tidx2,sidx2); end
                    if isfield(obj.(f),'resL'); dst.(f).resL(tidx1,sidx1) = obj.(f).resL(tidx2,sidx2); end
                    if isfield(obj.(f),'resD'); dst.(f).resD(tidx1,sidx1) = obj.(f).resD(tidx2,sidx2); end
                    if isfield(obj.(f),'resPc'); dst.(f).resPc(tidx1,sidx1) = obj.(f).resPc(tidx2,sidx2); end
                    if isfield(obj.(f),'resLc'); dst.(f).resLc(tidx1,sidx1) = obj.(f).resLc(tidx2,sidx2); end
                    if isfield(obj.(f),'resDc'); dst.(f).resDc(tidx1,sidx1) = obj.(f).resDc(tidx2,sidx2); end
                    if isfield(obj.(f),'Pd'); dst.(f).Pd(tidx1,sidx1) = obj.(f).Pd(tidx2,sidx2); end
                    if isfield(obj.(f),'Ld'); dst.(f).Ld(tidx1,sidx1) = obj.(f).Ld(tidx2,sidx2); end
                    if isfield(obj.(f),'Dd'); dst.(f).Dd(tidx1,sidx1) = obj.(f).Dd(tidx2,sidx2); end
                    if isfield(obj.(f),'resPd'); dst.(f).resPd(tidx1,sidx1) = obj.(f).resPd(tidx2,sidx2); end
                    if isfield(obj.(f),'resLd'); dst.(f).resLd(tidx1,sidx1) = obj.(f).resLd(tidx2,sidx2); end
                    if isfield(obj.(f),'resDd'); dst.(f).resDd(tidx1,sidx1) = obj.(f).resDd(tidx2,sidx2); end
                    if isfield(obj.(f),'Pdd'); dst.(f).Pdd(tidx1,sidx1) = obj.(f).Pdd(tidx2,sidx2); end
                    if isfield(obj.(f),'Ldd'); dst.(f).Ldd(tidx1,sidx1) = obj.(f).Ldd(tidx2,sidx2); end
                    if isfield(obj.(f),'resPdd'); dst.(f).resPdd(tidx1,sidx1) = obj.(f).resPdd(tidx2,sidx2); end
                    if isfield(obj.(f),'resLdd'); dst.(f).resLdd(tidx1,sidx1) = obj.(f).resLdd(tidx2,sidx2); end
                end
            end
        end
    end
end