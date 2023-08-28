function MoveRectangle(rec, center)
oldPosition = rec.Position;
rec.Position = [...
    center(1) - oldPosition(3)/2 ...
    center(2) - oldPosition(4)/2 ...
    oldPosition(3) ...
    oldPosition(4)];
end

