from __future__ import annotations

from transformers import LiltForTokenClassification, LayoutLMv3Processor, LayoutLMv3ImageProcessor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import functools
import numpy as np
from importlib import resources
from PIL import Image
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING: from ocr_microservice.ocr_pipeline.injector import Injector
from ocr_microservice.ocr_pipeline.post_ocr.helpers.label_groups import LabelGroups
from ocr_microservice.ocr_pipeline.resources import fonts



# sort by upperleft coordinate
def _compare_upperleft(tupleA,tupleB):
    (_,bboxA,_) = tupleA['ocr']
    (_,bboxB,_) = tupleB['ocr']
    [xA1,yA1,xA2,yA2] = bboxA
    [xB1,yB1,xB2,yB2] = bboxB

    if yA1 > yA2:
        return -1
    elif yA1 < yA2:
        return 1
    
    if xA1 > xA2:
        return -1
    if xA1 < xA2:
        return 1
    
    return 0

class PostProcessLilt:
    def __init__(self, labelGroups, liltResults) -> None:
        self.liltResults = liltResults
        self.labelGroups = labelGroups
    
    def sortByUpperLeft(self, liltResults):
        # sort by upperleft corner -> y coordinate is most important
        return sorted(liltResults,key=functools.cmp_to_key(_compare_upperleft))
    
    def buildLines(self, sortedLiltResults):
        # make sets which overlap by y coordinate
        lines = []
        line = []
        for t in sortedLiltResults:
            (_,bbox,_) = t['ocr']
            [_,y1,_,y2] = bbox
            if line:
                (_,prev_bbox,_) = line[-1]['ocr']
                [_,prev_y1,_,prev_y2] = prev_bbox
            elif lines:
                prev_line = lines[-1]
                (_,prev_bbox,_) = prev_line[-1]['ocr']
                [_,prev_y1,_,prev_y2] = prev_bbox
            else:
                prev_y1 = -1
                prev_y2 = -1

            new_line = False
            if prev_y1 == -1: # no previous item
                new_line = True
            elif y1 >= prev_y2: # clearly below previous item
                new_line = True
            elif y2 <= prev_y2: # cur item completely inside prev item
                new_line = False
            else: # in this case, y1 > prev_y1 and y2 > prev_y2
                full_range = y2 - prev_y1
                overlap = prev_y2 - y1
                if (overlap / full_range) < 0.50:
                    new_line = True

            # detect new line
            if new_line:
                if line:
                    lines.append(line)
                line = []
            
            line.append(t)

        if line:
            lines.append(line)

        return lines
    
    def getMergedLabels(self,line):
        merged_labels = {}
        # initialize merged_labels
        for key in self.labelGroups.getGroups():
            merged_labels[key] = 0
        
        for t in line:
            # merged_labels
            labels = t['lilt']
            for l in labels:
                (l_id,_,c) = l
                if l_id == 0: # ignore because if confidence is high then we don't want it to determine the whole line
                    continue
                group = self.labelGroups.getGroupByLabelId(l_id)
                # main concept: if a line has more items of the same group then the confidence that the group is correct increases
                # this is esp. true for product item rows
                merged_labels[group] += round((1-merged_labels[group])*c,3)
            
        merged_labels = list(sorted(merged_labels.items(), key=lambda item: item[1], reverse=True))[:3]

        return merged_labels
    
    # check whether conflicting labels are in the same line
    # e.g. an item description cannot be on the same line as a store address
    def hasLineConflict(self,line):
        group = None
        for t in line:
            labels = t['lilt']
            toplabel = labels[0]
            (l_id,_,_) = toplabel
            g = self.labelGroups.getGroupByLabelId(l_id)
            if g == 'O': # ignore cannot cause a conflict
                continue

            if group == None:
                group = g
            elif group != g:
                return True # group changes
        return False
    
    def getSingleLineInfo(self,i,line):
        info = {}
        info["line_no"] = i
        info["merged_labels"] = self.getMergedLabels(line)
        info["line_conflict"] = self.hasLineConflict(line)

        return info
    
    def addMetadataAllLines(self,lines):
        result = []
        i = 0
        for line in lines:
            result.append({"ocr_lilt": line, "line_info": self.getSingleLineInfo(i,line)})
            i += 1
        return result
    
    def calcLineBox(self, line):
        x_max = -1
        x_min = 10000000
        y_max = -1
        y_min = 10000000
        for entry in line:
            (_,bbox,_) = entry['ocr']
            [x1,y1,x2,y2] = bbox
            x_min = min(x_min,x1,x2)
            x_max = max(x_max,x1,x2)
            y_min = min(y_min,y1,y2)
            y_max = max(y_max,y1,y2)
            
        return [x_min,y_min,x_max,y_max]
    
    # draw results onto the image
    def draw_boxes(self, image: Image.Image, result) -> Image.Image:
        # draw predictions over the image
        draw = ImageDraw.Draw(image)
        
        font_path = str(resources.path(fonts, "DejaVuSans.ttf"))
        font = ImageFont.truetype(font_path)
        for r in result:
            line = r['ocr_lilt']
            info = r['line_info']
            merged_labels = info['merged_labels']
            group,confidence = merged_labels[0]
            line_conflict = info['line_conflict']
            bbox = self.calcLineBox(line)

            draw.rectangle(bbox, outline=self.labelGroups.getColor(group)) #label2color.label2color[ner])
            text = group + ' ' + str(confidence) + ' (conflict = ' + str(line_conflict) + ')'
            draw.text((bbox[0] + 10, bbox[1] - 10), text=text, fill=self.labelGroups.getColor(group), font=font)
        return image
    
    def run(self):
        sorted = self.sortByUpperLeft(self.liltResults)
        lines = self.buildLines(sorted)
        lines_with_metadata = self.addMetadataAllLines(lines)
        return lines_with_metadata
    
def post_process_lilt(image: Image.Image, data: dict, injector: Injector) -> Tuple[Image.Image, dict]:
    model = injector.models.lilt_model
    labelGroups = LabelGroups(model.config.id2label, model.config.label2id)
    postLilt = PostProcessLilt(labelGroups,data)
    result = postLilt.run()
    image = postLilt.draw_boxes(image,result)
    return image, {"ocr_lilt": data, "lines": result }
